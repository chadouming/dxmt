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

namespace {

// Coordinates, array slice and sampler choice for a texture kind, as airconv's DXBC sample lowering builds them
// (nt/dxbc_converter_base.cpp, InstSample). `c` holds DXIL's c0..c3 (undef where unused).
struct Addressing {
  llvm::Value *coord = nullptr;
  llvm::Value *array = nullptr;
  bool cube = false;
};

bool IsArray(Texture::ResourceKind k) {
  switch (k) {
  case Texture::texture1d_array: case Texture::texture2d_array: case Texture::depth2d_array: case Texture::texturecube_array:
  case Texture::depthcube_array: case Texture::texture2d_ms_array: case Texture::depth_2d_ms_array: return true;
  default: return false;
  }
}

bool IsMultisampled(Texture::ResourceKind k) {
  return k == Texture::texture2d_ms || k == Texture::texture2d_ms_array || k == Texture::depth_2d_ms ||
         k == Texture::depth_2d_ms_array;
}

// Components of the position/coordinate vector (1D textures are 2D in Metal here, as the DXBC path treats them).
unsigned Dimensions(Texture::ResourceKind k) {
  switch (k) {
  case Texture::texture3d: case Texture::texturecube: case Texture::depthcube: case Texture::texturecube_array:
  case Texture::depthcube_array: return 3;
  default: return 2;
  }
}

// The component of c0..c3 holding the array slice.
unsigned ArrayComponent(Texture::ResourceKind k) {
  switch (k) {
  case Texture::texture1d_array: return 1;
  case Texture::texturecube_array: case Texture::depthcube_array: return 3;
  default: return 2;
  }
}

} // namespace

llvm::Error
Lowering::LowerTexture(uint32_t opcode, llvm::CallInst *call, const HandleInfo &h) {
  auto arg = [&](unsigned i) { return call->getArgOperand(i); };
  auto ty = call->getType();
  auto d = Texture(h);
  if (!d) {
    Replace(call, ty->isVoidTy() ? nullptr : llvm::Constant::getNullValue(ty));
    return llvm::Error::success();
  }
  llvm::air::Texture tex;
  tex.kind = d->ResourceKind;
  tex.sample_type = d->SampleType;
  tex.memory_access = d->MemoryAccess;
  auto logical = d->ResourceKindLogical;
  unsigned dims = Dimensions(logical);
  bool one_d = logical == Texture::texture1d || logical == Texture::texture1d_array;

  // Float coordinates from c0.. (sample/gather/lod) or integer positions (load/store).
  auto vector = [&](unsigned first, llvm::Type *elem) -> llvm::Value * {
    llvm::Value *v = llvm::UndefValue::get(llvm::FixedVectorType::get(elem, dims));
    for (unsigned i = 0; i < dims; i++)
      v = ir.CreateInsertElement(v, one_d && i == 1 ? llvm::Constant::getNullValue(elem) : arg(first + i), i);
    return v;
  };
  auto array_index = [&](unsigned first) -> llvm::Value * {
    if (!IsArray(logical))
      return nullptr;
    auto v = arg(first + ArrayComponent(logical));
    return v->getType()->isFloatingPointTy() ? conv.ClampArrayIndex(v, d->Metadata) : v;
  };
  auto offsets = [&](unsigned first, unsigned count, int32_t out[3]) {
    for (unsigned i = 0; i < 3; i++)
      out[i] = i < count ? (int32_t)ConstantU32(arg(first + i), 0) : 0;
  };
  // The texel (vector or depth scalar) as the call's ResRet struct.
  auto result = [&](llvm::Value *texel) {
    auto elem = ty->getStructElementType(0);
    llvm::Value *fields[5];
    for (unsigned c = 0; c < 4; c++) {
      auto v = texel->getType()->isVectorTy() ? ir.CreateExtractElement(texel, c) : texel;
      fields[c] = Cast(v, elem);
    }
    fields[4] = ir.getInt32(1); // status: fully mapped
    Replace(call, Aggregate(ty, fields));
    return llvm::Error::success();
  };

  switch (opcode) {
  case op::TextureLoad: { // (66, handle, i32 mipOrSample, i32 c0, c1, c2, i32 o0, o1, o2)
    auto pos = vector(3, ir.getInt32Ty());
    int32_t o[3];
    offsets(6, dims, o);
    llvm::Value *offset = llvm::UndefValue::get(pos->getType());
    for (unsigned i = 0; i < dims; i++)
      offset = ir.CreateInsertElement(offset, ir.getInt32(o[i]), i);
    pos = ir.CreateAdd(pos, offset);
    bool ms = IsMultisampled(logical);
    if (tex.memory_access == Texture::acesss_readwrite) // see earlier writes, as the DXBC typed UAV load
      air.CreateTextureFence(tex, d->ResourceHandle);
    auto [texel, residency] = air.CreateRead(tex, d->ResourceHandle, pos, array_index(3), ms ? arg(2) : nullptr,
                                             ms ? ir.getInt32(0) : arg(2), d->GlobalCoherent);
    return result(texel);
  }
  case op::TextureStore: { // (67, handle, i32 c0, c1, c2, v0, v1, v2, v3, i8 mask)
    auto pos = vector(2, ir.getInt32Ty());
    auto texel_ty = llvm::cast<llvm::FixedVectorType>(air.getTexelType(tex));
    llvm::Value *texel = llvm::UndefValue::get(texel_ty);
    for (unsigned c = 0; c < 4; c++)
      texel = ir.CreateInsertElement(texel, Cast(arg(5 + c), texel_ty->getElementType()), c);
    air.CreateWrite(tex, d->ResourceHandle, pos, array_index(2), nullptr, ir.getInt32(0), texel, d->GlobalCoherent);
    Replace(call, nullptr);
    return llvm::Error::success();
  }
  default:
    break;
  }

  // Sampling and gathering: the sampler is argument 2, coordinates start at 3.
  auto sampler_handle = ResolveHandle(entry, arg(2));
  if (!sampler_handle)
    return sampler_handle.takeError();
  auto s = ctx.binding.GetSampler(air, sampler_handle->range, sampler_handle->index);
  if (!s) {
    Replace(call, llvm::Constant::getNullValue(ty));
    return llvm::Error::success();
  }
  auto sampler = Dimensions(logical) == 3 && logical != Texture::texture3d ? s->CubeSamplerHandle : s->SamplerHandle;
  auto bias = ir.CreateBitCast(ir.CreateTrunc(s->Metadata, ir.getInt32Ty()), ir.getFloatTy()); // the sampler's LOD bias
  auto coord = vector(3, ir.getFloatTy());
  auto array = array_index(3);
  int32_t o[3];
  offsets(7, dims, o);
  auto min_lod = [&](llvm::Value *shader_clamp) {
    auto clamp = conv.DecodeTextureMinLODClamp(d->Metadata);
    if (shader_clamp && !llvm::isa<llvm::UndefValue>(shader_clamp))
      clamp = air.CreateFPBinOp(llvm::air::AIRBuilder::fmax, clamp, shader_clamp);
    return llvm::air::sample_min_lod_clamp{clamp};
  };

  switch (opcode) {
  case op::Sample: // (60, srv, sampler, c0..c3, o0..o2, clamp)
    return result(air.CreateSample(tex, d->ResourceHandle, sampler, coord, array, o, llvm::air::sample_bias{bias}, min_lod(arg(10))).first);
  case op::SampleBias: // (61, ..., bias, clamp)
    return result(air.CreateSample(tex, d->ResourceHandle, sampler, coord, array, o,
                                   llvm::air::sample_bias{ir.CreateFAdd(arg(10), bias)}, min_lod(arg(11))).first);
  case op::SampleLevel: // (62, ..., lod)
    return result(air.CreateSample(tex, d->ResourceHandle, sampler, coord, array, o,
                                   llvm::air::sample_level{ir.CreateFAdd(arg(10), bias)}).first);
  case op::SampleGrad: { // (63, ..., ddx0..2, ddy0..2, clamp)
    llvm::Value *ddx = llvm::UndefValue::get(coord->getType()), *ddy = ddx;
    for (unsigned i = 0; i < dims; i++) {
      ddx = ir.CreateInsertElement(ddx, one_d && i == 1 ? llvm::ConstantFP::get(ir.getFloatTy(), 0) : arg(10 + i), i);
      ddy = ir.CreateInsertElement(ddy, one_d && i == 1 ? llvm::ConstantFP::get(ir.getFloatTy(), 0) : arg(13 + i), i);
    }
    return result(air.CreateSampleGrad(tex, d->ResourceHandle, sampler, coord, array, ddx, ddy, min_lod(arg(16)).lod, o).first);
  }
  case op::SampleCmp: // (64, ..., compare, clamp)
    return result(air.CreateSampleCmp(tex, d->ResourceHandle, sampler, coord, array, arg(10), o,
                                      llvm::air::sample_bias{bias}, min_lod(arg(11))).first);
  case op::SampleCmpLevelZero: // (65, ..., compare)
    return result(air.CreateSampleCmp(tex, d->ResourceHandle, sampler, coord, array, arg(10), o,
                                      llvm::air::sample_level{llvm::ConstantFP::get(ir.getFloatTy(), 0)}).first);
  case op::TextureGather: { // (73, srv, sampler, c0..c3, i32 o0, o1, i32 channel)
    int32_t go[3];
    offsets(7, 2, go);
    return result(air.CreateGather(tex, d->ResourceHandle, sampler, coord, array, go, arg(9)).first);
  }
  case op::TextureGatherCmp: { // (74, ..., i32 channel, float compare)
    int32_t go[3];
    offsets(7, 2, go);
    return result(air.CreateGatherCompare(tex, d->ResourceHandle, sampler, coord, array, arg(10), go).first);
  }
  case op::CalculateLOD: { // (81, srv, sampler, c0, c1, c2, i1 clamped)
    auto lod = air.CreateCalculateLOD(tex, d->ResourceHandle, sampler, coord);
    Replace(call, ConstantU32(arg(6), 1) ? lod.first : lod.second);
    return llvm::Error::success();
  }
  default:
    return Unsupported(call);
  }
}

// %dx.types.Dimensions {width, height, depth or array size, mip levels or sample count}, as the DXBC resinfo/bufinfo
// lowerings answer them.
llvm::Error
Lowering::LowerDimensions(llvm::CallInst *call, const HandleInfo &h) {
  auto ty = call->getType();
  auto kind = h.resource->kind;
  if (kind == ResourceKind::RawBuffer || kind == ResourceKind::StructuredBuffer) {
    auto buffer = Buffer(h);
    llvm::Value *size = buffer && buffer->Metadata ? conv.DecodeRawBufferByteLength(buffer->Metadata) : ir.getInt32(0);
    if (kind == ResourceKind::StructuredBuffer && h.resource->stride)
      size = ir.CreateUDiv(size, ir.getInt32(h.resource->stride));
    Replace(call, Aggregate(ty, {size, ir.getInt32(0), ir.getInt32(0), ir.getInt32(0)}));
    return llvm::Error::success();
  }
  auto d = Texture(h);
  if (!d) {
    Replace(call, llvm::Constant::getNullValue(ty));
    return llvm::Error::success();
  }
  llvm::air::Texture tex;
  tex.kind = d->ResourceKind;
  tex.sample_type = d->SampleType;
  tex.memory_access = d->MemoryAccess;
  auto logical = d->ResourceKindLogical;
  if (kind == ResourceKind::TypedBuffer) {
    Replace(call, Aggregate(ty, {conv.DecodeTextureBufferElement(d->Metadata), ir.getInt32(0), ir.getInt32(0), ir.getInt32(0)}));
    return llvm::Error::success();
  }
  auto level = call->getArgOperand(2);
  if (llvm::isa<llvm::UndefValue>(level))
    level = ir.getInt32(0);
  auto q = [&](Texture::Query query, llvm::Value *lod) { return air.CreateTextureQuery(tex, d->ResourceHandle, query, lod); };
  llvm::Value *x = q(Texture::width, level), *y = ir.getInt32(0), *z = ir.getInt32(0);
  if (logical == Texture::texture1d_array)
    y = q(Texture::array_length, ir.getInt32(0));
  else if (logical != Texture::texture1d)
    y = q(Texture::height, level);
  if (logical == Texture::texture3d)
    z = q(Texture::depth, level);
  else if (IsArray(logical) && logical != Texture::texture1d_array)
    z = q(Texture::array_length, ir.getInt32(0));
  auto last = IsMultisampled(logical) ? q(Texture::num_samples, ir.getInt32(0)) : q(Texture::num_mip_levels, ir.getInt32(0));
  Replace(call, Aggregate(ty, {x, y, z, last}));
  return llvm::Error::success();
}

llvm::Error
Lowering::LowerResource(uint32_t opcode, llvm::CallInst *call) {
  if (opcode == op::CheckAccessFullyMapped) { // (71, i32 status): every resource is fully mapped here
    Replace(call, ir.CreateICmpNE(call->getArgOperand(1), ir.getInt32(0)));
    return llvm::Error::success();
  }
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
  case op::GetDimensions: // (72, handle, i32 mipLevel)
    return LowerDimensions(call, h);
  case op::TextureLoad:
  case op::TextureStore:
  case op::Sample:
  case op::SampleBias:
  case op::SampleLevel:
  case op::SampleGrad:
  case op::SampleCmp:
  case op::SampleCmpLevelZero:
  case op::TextureGather:
  case op::TextureGatherCmp:
  case op::CalculateLOD:
    return LowerTexture(opcode, call, h);
  default:
    return Unsupported(call);
  }
}

} // namespace dxmt::dxil
