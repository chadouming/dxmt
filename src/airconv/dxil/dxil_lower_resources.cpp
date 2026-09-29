/*
 * Copyright 2026 MacNeutron contributors
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA
 */
#include "dxil_lower.hpp"
#include "../airconv_error.hpp"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"

namespace dxmt::dxil {

using llvm::air::Texture;

llvm::Optional<dxbc::BufferResourceHandle>
Lowering::Buffer(const HandleInfo &h) {
  auto d = h.cls == ResourceClass::UAV ? ctx.binding.GetUAVBuffer(air, h.range, h.index)
                                       : ctx.binding.GetSRVBuffer(air, h.range, h.index);
  if (!d)
    return {};
  return dxbc::BufferResourceHandle{d->Pointer, d->Metadata, d->StructureStride, dxbc::swizzle_identity, d->GlobalCoherent};
}

llvm::Optional<dxbc::TextureDescirptor>
Lowering::Texture(const HandleInfo &h) {
  return h.cls == ResourceClass::UAV ? ctx.binding.GetUAVTexture(air, h.range, h.index)
                                     : ctx.binding.GetSRVTexture(air, h.range, h.index);
}

llvm::Value *
Lowering::ElementPointer(dxbc::BufferResourceHandle &buffer, llvm::Value *byte_offset, llvm::Type *ty) {
  unsigned space = llvm::cast<llvm::PointerType>(buffer.Pointer->getType())->getAddressSpace();
  auto bytes = ir.CreateBitCast(buffer.Pointer, ir.getInt8PtrTy(space));
  auto ptr = ir.CreateBitCast(ir.CreateGEP(ir.getInt8Ty(), bytes, byte_offset), ty->getPointerTo(space));
  if (!buffer.Metadata)
    return ptr;
  auto end = ir.CreateAdd(byte_offset, ir.getInt32(ty->getPrimitiveSizeInBits() / 8));
  auto in_bounds = ir.CreateICmpULE(end, conv.DecodeRawBufferByteLength(buffer.Metadata));
  return ir.CreateSelect(in_bounds, ptr, llvm::Constant::getNullValue(ptr->getType()));
}

llvm::Value *
Lowering::LoadElement(dxbc::BufferResourceHandle &buffer, llvm::Value *byte_offset, llvm::Type *ty) {
  auto ptr = ElementPointer(buffer, byte_offset, ty);
  return buffer.GlobalCoherent ? (llvm::Value *)air.CreateDeviceCoherentLoad(ty, ptr) : ir.CreateLoad(ty, ptr);
}

void
Lowering::StoreElement(dxbc::BufferResourceHandle &buffer, llvm::Value *byte_offset, llvm::Value *value) {
  auto ptr = ElementPointer(buffer, byte_offset, value->getType());
  if (buffer.GlobalCoherent)
    air.CreateDeviceCoherentStore(value, ptr);
  else
    ir.CreateStore(value, ptr);
}

// %dx.types.ResRet.T = {T, T, T, T, i32 status}; components outside `mask` stay undef.
llvm::Error
Lowering::LowerBufferLoad(llvm::CallInst *call, const HandleInfo &h, llvm::Value *byte_offset, uint32_t mask) {
  auto buffer = Buffer(h);
  auto ty = call->getType();
  auto elem = ty->getStructElementType(0);
  if (!buffer) {
    Replace(call, llvm::Constant::getNullValue(ty));
    return llvm::Error::success();
  }
  llvm::Value *fields[5] = {};
  unsigned size = elem->getPrimitiveSizeInBits() / 8;
  for (unsigned c = 0; c < 4; c++)
    if (mask & (1 << c))
      fields[c] = LoadElement(*buffer, ir.CreateAdd(byte_offset, ir.getInt32(c * size)), elem);
  fields[4] = ir.getInt32(1); // status: fully mapped
  Replace(call, Aggregate(ty, fields));
  return llvm::Error::success();
}

llvm::Error
Lowering::LowerBufferStore(llvm::CallInst *call, const HandleInfo &h, llvm::Value *byte_offset, unsigned first_value, uint32_t mask) {
  auto buffer = Buffer(h);
  if (buffer) {
    auto elem = call->getArgOperand(first_value)->getType();
    unsigned size = elem->getPrimitiveSizeInBits() / 8;
    for (unsigned c = 0; c < 4; c++)
      if (mask & (1 << c))
        StoreElement(*buffer, ir.CreateAdd(byte_offset, ir.getInt32(c * size)), call->getArgOperand(first_value + c));
  }
  Replace(call, nullptr);
  return llvm::Error::success();
}

// Typed buffers are Metal texture buffers; the descriptor's metadata holds the view's first element and element count
// (as the DXBC ld/store_uav_typed lowering reads it).
llvm::Error
Lowering::LowerTypedBufferLoad(llvm::CallInst *call, const HandleInfo &h, llvm::Value *index) {
  auto d = Texture(h);
  auto ty = call->getType();
  if (!d) {
    Replace(call, llvm::Constant::getNullValue(ty));
    return llvm::Error::success();
  }
  llvm::air::Texture tex;
  tex.kind = d->ResourceKind;
  tex.sample_type = d->SampleType;
  tex.memory_access = d->MemoryAccess;
  auto oob = ir.CreateICmpUGE(index, conv.DecodeTextureBufferElement(d->Metadata));
  auto address = ir.CreateAdd(index, conv.DecodeTextureBufferOffset(d->Metadata));
  address = ir.CreateSelect(oob, llvm::ConstantInt::getAllOnesValue(address->getType()), address);
  if (tex.memory_access == llvm::air::Texture::acesss_readwrite) // as the DXBC typed UAV load: see earlier writes
    air.CreateTextureFence(tex, d->ResourceHandle);
  auto [texel, residency] = air.CreateRead(tex, d->ResourceHandle, address, nullptr, nullptr, ir.getInt32(0), d->GlobalCoherent);
  auto elem = ty->getStructElementType(0);
  llvm::Value *fields[5];
  for (unsigned c = 0; c < 4; c++)
    fields[c] = Cast(ir.CreateExtractElement(texel, c), elem);
  fields[4] = ir.getInt32(1);
  Replace(call, Aggregate(ty, fields));
  return llvm::Error::success();
}

llvm::Error
Lowering::LowerTypedBufferStore(llvm::CallInst *call, const HandleInfo &h, llvm::Value *index) {
  // bufferStore(69, handle, index, offset, v0, v1, v2, v3, mask) / textureStore-like: values at 4..7
  auto d = Texture(h);
  if (d) {
    llvm::air::Texture tex;
    tex.kind = d->ResourceKind;
    tex.sample_type = d->SampleType;
    tex.memory_access = d->MemoryAccess;
    auto texel_ty = llvm::cast<llvm::FixedVectorType>(air.getTexelType(tex));
    llvm::Value *texel = llvm::UndefValue::get(texel_ty);
    for (unsigned c = 0; c < 4; c++)
      texel = ir.CreateInsertElement(texel, Cast(call->getArgOperand(4 + c), texel_ty->getElementType()), c);
    auto address = ir.CreateAdd(index, conv.DecodeTextureBufferOffset(d->Metadata));
    air.CreateWrite(tex, d->ResourceHandle, address, nullptr, nullptr, ir.getInt32(0), texel, d->GlobalCoherent);
  }
  Replace(call, nullptr);
  return llvm::Error::success();
}

llvm::Error
Lowering::LowerResource(uint32_t opcode, llvm::CallInst *call) {
  auto handle = ResolveHandle(entry, call->getArgOperand(1));
  if (!handle)
    return handle.takeError();
  auto &h = *handle;
  auto kind = h.resource->kind;
  bool structured = kind == ResourceKind::StructuredBuffer;
  auto stride = ir.getInt32(h.resource->stride);

  switch (opcode) {
  case op::CBufferLoadLegacy: { // (59, handle, i32 row) -> %dx.types.CBufRet.T
    auto ty = call->getType();
    auto d = ctx.binding.GetConstantBuffer(air, h.range, h.index);
    if (!d) {
      Replace(call, llvm::Constant::getNullValue(ty));
      return llvm::Error::success();
    }
    auto row_ty = air.getIntTy(4);
    auto row = ir.CreateLoad(row_ty, ir.CreateGEP(row_ty, d->Pointer, {call->getArgOperand(2)}));
    auto elem = ty->getStructElementType(0);
    unsigned count = ty->getStructNumElements(), bits = elem->getPrimitiveSizeInBits();
    llvm::SmallVector<llvm::Value *, 8> fields;
    for (unsigned e = 0; e < count; e++) {
      llvm::Value *v;
      if (bits == 32) {
        v = ir.CreateExtractElement(row, e);
      } else if (bits == 16) {
        v = ir.CreateTrunc(ir.CreateLShr(ir.CreateExtractElement(row, e / 2), (e % 2) * 16), ir.getInt16Ty());
      } else { // 64
        auto lo = ir.CreateZExt(ir.CreateExtractElement(row, e * 2), ir.getInt64Ty());
        auto hi = ir.CreateZExt(ir.CreateExtractElement(row, e * 2 + 1), ir.getInt64Ty());
        v = ir.CreateOr(lo, ir.CreateShl(hi, 32));
      }
      fields.push_back(ir.CreateBitCast(v, elem));
    }
    Replace(call, Aggregate(ty, fields));
    return llvm::Error::success();
  }
  case op::CBufferLoad: { // (58, handle, i32 byteOffset, i32 alignment) -> T
    auto ty = call->getType();
    auto d = ctx.binding.GetConstantBuffer(air, h.range, h.index);
    if (!d) {
      Replace(call, llvm::Constant::getNullValue(ty));
      return llvm::Error::success();
    }
    auto offset = call->getArgOperand(2);
    auto word_ptr = ir.CreateGEP(air.getIntTy(4), d->Pointer, {ir.CreateLShr(offset, 4), ir.CreateAnd(ir.CreateLShr(offset, 2), 3)});
    llvm::Value *word = ir.CreateLoad(ir.getInt32Ty(), word_ptr);
    if (ty->getPrimitiveSizeInBits() == 16)
      word = ir.CreateTrunc(ir.CreateLShr(word, ir.CreateShl(ir.CreateAnd(offset, 2), 3)), ir.getInt16Ty());
    else if (ty->getPrimitiveSizeInBits() == 64)
      return llvm::make_error<UnsupportedFeature>("DXIL: 64-bit cbufferLoad not supported");
    Replace(call, ir.CreateBitCast(word, ty));
    return llvm::Error::success();
  }
  case op::RawBufferLoad: { // (139, handle, i32 index, i32 elementOffset, i8 mask, i32 alignment)
    auto offset = structured ? ir.CreateAdd(ir.CreateMul(call->getArgOperand(2), stride), call->getArgOperand(3))
                             : call->getArgOperand(2);
    return LowerBufferLoad(call, h, offset, ConstantU32(call->getArgOperand(4), 0xf));
  }
  case op::RawBufferStore: { // (140, handle, i32 index, i32 elementOffset, v0..v3, i8 mask, i32 alignment)
    auto offset = structured ? ir.CreateAdd(ir.CreateMul(call->getArgOperand(2), stride), call->getArgOperand(3))
                             : call->getArgOperand(2);
    return LowerBufferStore(call, h, offset, 4, ConstantU32(call->getArgOperand(8), 0xf));
  }
  case op::BufferLoad: // (68, handle, i32 index, i32 offset)
    if (kind == ResourceKind::TypedBuffer)
      return LowerTypedBufferLoad(call, h, call->getArgOperand(2));
    return LowerBufferLoad(call, h,
                           structured ? ir.CreateAdd(ir.CreateMul(call->getArgOperand(2), stride), call->getArgOperand(3))
                                      : call->getArgOperand(2),
                           0xf);
  case op::BufferStore: // (69, handle, i32 index, i32 offset, v0..v3, i8 mask)
    if (kind == ResourceKind::TypedBuffer)
      return LowerTypedBufferStore(call, h, call->getArgOperand(2));
    return LowerBufferStore(call, h,
                            structured ? ir.CreateAdd(ir.CreateMul(call->getArgOperand(2), stride), call->getArgOperand(3))
                                       : call->getArgOperand(2),
                            4, ConstantU32(call->getArgOperand(8), 0xf));
  default:
    return Unsupported(call);
  }
}

} // namespace dxmt::dxil
