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

uint32_t
ConstantU32(llvm::Value *v, uint32_t otherwise) {
  auto c = llvm::dyn_cast<llvm::ConstantInt>(v);
  return c ? (uint32_t)c->getZExtValue() : otherwise;
}

llvm::Expected<HandleInfo>
ResolveHandle(const EntryInfo &entry, llvm::Value *handle) {
  auto call = llvm::dyn_cast<llvm::CallInst>(handle);
  if (!call)
    return llvm::make_error<UnsupportedFeature>("DXIL: resource handle through phi/select");
  switch (OpCode(*call)) {
  case op::AnnotateHandle: // (216, handle, %dx.types.ResourceProperties)
    return ResolveHandle(entry, call->getArgOperand(1));
  case op::CreateHandle: { // (57, i8 class, i32 rangeId, i32 index, i1 nonUniform)
    auto cls = ResourceClass(ConstantU32(call->getArgOperand(1)));
    uint32_t id = ConstantU32(call->getArgOperand(2));
    auto r = entry.FindResource(cls, id);
    if (!r)
      return llvm::make_error<UnsupportedFeature>("DXIL: createHandle names an undeclared resource");
    return HandleInfo{cls, id, call->getArgOperand(3), r};
  }
  case op::CreateHandleFromBinding: { // (217, %dx.types.ResBind {i32 LB, i32 UB, i32 space, i8 class}, i32 index, i1)
    auto bind = llvm::dyn_cast<llvm::Constant>(call->getArgOperand(1));
    if (!bind)
      return llvm::make_error<UnsupportedFeature>("DXIL: non-constant createHandleFromBinding");
    auto field = [&](unsigned i) { return ConstantU32(bind->getAggregateElement(i)); };
    uint32_t lb = field(0), space = field(2);
    auto cls = ResourceClass(field(3));
    for (auto &r : entry.resources)
      if (r.cls == cls && r.space == space && lb >= r.lower_bound && (r.size == ~0u || lb - r.lower_bound < r.size))
        return HandleInfo{cls, r.id, call->getArgOperand(2), &r};
    return llvm::make_error<UnsupportedFeature>("DXIL: createHandleFromBinding names an undeclared resource");
  }
  case op::CreateHandleFromHeap:
    return llvm::make_error<UnsupportedFeature>("DXIL: dx.op.createHandleFromHeap (dynamic resources) not supported");
  default:
    return llvm::make_error<UnsupportedFeature>("DXIL: resource handle from an unrecognised op");
  }
}

Lowering::Lowering(const EntryInfo &entry, dxbc::context &ctx) :
    entry(entry),
    ctx(ctx),
    ir(ctx.builder),
    air(ctx.air),
    conv(ctx.air, ctx, ctx.resource) {}

llvm::Error
Lowering::Unsupported(llvm::CallInst *call) {
  return llvm::make_error<UnsupportedFeature>(
      "DXIL: " + call->getCalledFunction()->getName().str() + " (" + std::to_string(OpCode(*call)) + ") not supported"
  );
}

void
Lowering::Replace(llvm::CallInst *call, llvm::Value *value) {
  if (value)
    call->replaceAllUsesWith(value);
  call->eraseFromParent();
}

llvm::Value *
Lowering::Aggregate(llvm::Type *ty, llvm::ArrayRef<llvm::Value *> fields) {
  llvm::Value *agg = llvm::UndefValue::get(ty);
  for (unsigned i = 0; i < fields.size(); i++)
    if (fields[i])
      agg = ir.CreateInsertValue(agg, fields[i], {i});
  return agg;
}

llvm::Value *
Lowering::Cast(llvm::Value *value, llvm::Type *ty) {
  auto from = value->getType();
  if (from == ty)
    return value;
  if (from->getPrimitiveSizeInBits() == ty->getPrimitiveSizeInBits())
    return ir.CreateBitCast(value, ty);
  if (from->isFloatingPointTy() && ty->isFloatingPointTy())
    return ir.CreateFPCast(value, ty);
  if (from->isIntegerTy() && ty->isIntegerTy())
    return ir.CreateZExtOrTrunc(value, ty);
  if (from->isFloatingPointTy()) // float -> narrower/wider int: through the float's own-width int
    return ir.CreateZExtOrTrunc(ir.CreateBitCast(value, ir.getIntNTy(from->getPrimitiveSizeInBits())), ty);
  // int -> float of another width: resize, then reinterpret
  return ir.CreateBitCast(ir.CreateZExtOrTrunc(value, ir.getIntNTy(ty->getPrimitiveSizeInBits())), ty);
}

llvm::Error
Lowering::Lower(llvm::CallInst *call) {
  uint32_t opcode = OpCode(*call);
  switch (opcode) {
  case op::CreateHandle:
  case op::CreateHandleFromBinding:
  case op::AnnotateHandle:
    return llvm::Error::success(); // resolved at each use, erased by Cleanup
  case op::CreateHandleFromHeap:
    return ResolveHandle(entry, call).takeError();
  default:
    break;
  }
  ir.SetInsertPoint(call);
  if ((opcode >= op::FAbs && opcode <= op::Dot4) || opcode == op::MakeDouble || opcode == op::SplitDouble ||
      (opcode >= op::BitcastI16toF16 && opcode <= op::LegacyF16ToF32) ||
      (opcode >= op::Dot2AddHalf && opcode <= op::Dot4AddU8Packed))
    return LowerMath(opcode, call);
  if ((opcode >= op::CBufferLoad && opcode <= op::AtomicCompareExchange) || opcode == op::CalculateLOD ||
      opcode == op::RawBufferLoad || opcode == op::RawBufferStore)
    return LowerResource(opcode, call);
  return LowerOther(opcode, call);
}

void
Lowering::Cleanup(llvm::Function &function) {
  bool erased = true;
  while (erased) { // annotateHandle uses createHandle*: repeat until nothing changes
    erased = false;
    for (auto &bb : function)
      for (auto it = bb.begin(); it != bb.end();) {
        auto call = llvm::dyn_cast<llvm::CallInst>(&*it++);
        if (!call || !call->use_empty())
          continue;
        switch (OpCode(*call)) {
        case op::CreateHandle:
        case op::CreateHandleFromBinding:
        case op::AnnotateHandle:
          call->eraseFromParent();
          erased = true;
          break;
        }
      }
  }
}

llvm::Error
Lowering::LowerOther(uint32_t opcode, llvm::CallInst *call) {
  auto &res = ctx.resource;
  auto component = [&](llvm::Value *vec) {
    return vec ? ir.CreateExtractElement(vec, call->getArgOperand(1)) : nullptr;
  };
  switch (opcode) {
  case op::ThreadId: // (93, i32 component)
    Replace(call, component(res.thread_id_arg));
    return llvm::Error::success();
  case op::GroupId:
    Replace(call, component(res.thread_group_id_arg));
    return llvm::Error::success();
  case op::ThreadIdInGroup:
    Replace(call, component(res.thread_id_in_group_arg));
    return llvm::Error::success();
  case op::FlattenedThreadIdInGroup:
    Replace(call, res.thread_id_in_group_flat_arg);
    return llvm::Error::success();
  case op::Barrier: { // (80, i32 mode): 1 = sync threadgroup, 2 = UAV fence global, 4 = UAV fence threadgroup, 8 = TGSM fence
    // As airconv's DXBC sync lowering (nt/dxbc_converter_base.hpp, InstSync).
    using namespace llvm::air;
    uint32_t mode = ConstantU32(call->getArgOperand(1));
    bool sync = mode & 1, non_execution = ctx.metal_version >= SM50_SHADER_METAL_320;
    MemFlags flags = (mode & 8) ? MemFlags::Threadgroup : MemFlags::None;
    if (mode & 6) {
      flags |= MemFlags::Device | MemFlags::Texture;
      if (non_execution)
        air.CreateAtomicFence(flags, (mode & 2) ? (ThreadScope::Device | ThreadScope::Threadgroup) : ThreadScope::Threadgroup);
      if (sync)
        air.CreateBarrier(non_execution ? MemFlags::None : flags);
    } else if (sync) {
      air.CreateBarrier(flags);
    }
    Replace(call, nullptr);
    return llvm::Error::success();
  }
  default:
    return Unsupported(call);
  }
}

} // namespace dxmt::dxil
