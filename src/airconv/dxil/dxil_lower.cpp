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
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

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

Lowering::Lowering(const DXILShader &dxil, dxbc::context &ctx) :
    dxil(dxil),
    entry(dxil.entry),
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
      (opcode >= op::Dot2AddHalf && opcode <= op::Dot4AddU8Packed) || opcode == op::Unpack4x8 || opcode == op::Pack4x8)
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

llvm::Value *
Lowering::UnpackedInput(uint32_t element_id) {
  auto it = dxil.unpacked_inputs.find(element_id);
  if (it == dxil.unpacked_inputs.end())
    return nullptr;
  if (it->second == kUnpackedVertexID)
    return ctx.resource.vertex_id;
  if (it->second == kUnpackedInstanceID)
    return ctx.resource.instance_id;
  return ctx.function->getArg(it->second);
}

llvm::Value *
Lowering::UnpackedInputOfKind(SemanticKind kind) {
  for (auto &e : entry.inputs)
    if (e.kind == kind)
      return UnpackedInput(e.id);
  return nullptr;
}

llvm::Value *
Lowering::RegisterElementPointer(dxbc::register_file &file, const SignatureElement &e, llvm::Value *row, uint32_t col) {
  auto array_ty = llvm::cast<llvm::PointerType>(file.ptr_int4->getType())->getNonOpaquePointerElementType();
  auto reg = ir.CreateAdd(ir.getInt32(e.start_row), row);
  return ir.CreateGEP(array_ty, file.ptr_int4, {ir.getInt32(0), reg, ir.getInt32(e.start_col + col)});
}

namespace {

// Calls a Metal simdgroup function (AIR names as `xcrun metal -S -emit-llvm` shows them), marked convergent like Metal's.
llvm::Value *Simd(llvm::IRBuilder<> &ir, llvm::StringRef name, llvm::Type *ret, llvm::ArrayRef<llvm::Value *> args) {
  auto &context = ir.getContext();
  auto attrs = llvm::AttributeList::get(context, llvm::AttributeList::FunctionIndex,
                                        {llvm::Attribute::Convergent, llvm::Attribute::MustProgress,
                                         llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn});
  llvm::SmallVector<llvm::Type *, 2> types;
  for (auto a : args)
    types.push_back(a->getType());
  auto module = ir.GetInsertBlock()->getModule();
  auto fn = module->getOrInsertFunction(name, llvm::FunctionType::get(ret, types, false), attrs);
  auto call = ir.CreateCall(fn, args);
  call->setConvergent();
  return call;
}

// ".f32", ".f16", ".s.i32", ".u.i16", ... for a simdgroup function's value type.
std::string SimdSuffix(llvm::Type *ty, bool is_signed) {
  if (ty->isFloatTy()) return ".f32";
  if (ty->isHalfTy()) return ".f16";
  return std::string(is_signed ? ".s" : ".u") + ".i" + std::to_string(ty->getIntegerBitWidth());
}

const SignatureElement *FindElement(const std::vector<SignatureElement> &elements, uint32_t id) {
  for (auto &e : elements)
    if (e.id == id)
      return &e;
  return nullptr;
}
} // namespace

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
  case op::LoadInput: { // (4, i32 inputSigId, i32 row, i8 col, i32 gsVertexAxis)
    auto e = FindElement(entry.inputs, ConstantU32(call->getArgOperand(1)));
    if (!e)
      return llvm::make_error<UnsupportedFeature>("DXIL: loadInput names an undeclared input");
    if (auto v = UnpackedInput(e->id)) {
      Replace(call, Cast(v, call->getType()));
      return llvm::Error::success();
    }
    auto word = ir.CreateLoad(ir.getInt32Ty(), RegisterElementPointer(res.input, *e, call->getArgOperand(2),
                                                                     ConstantU32(call->getArgOperand(3))));
    auto ty = call->getType();
    Replace(call, ty->isIntegerTy(1) ? ir.CreateICmpNE(word, ir.getInt32(0)) : Cast(word, ty));
    return llvm::Error::success();
  }
  case op::StoreOutput: { // (5, i32 outputSigId, i32 row, i8 col, T value)
    auto e = FindElement(entry.outputs, ConstantU32(call->getArgOperand(1)));
    if (!e)
      return llvm::make_error<UnsupportedFeature>("DXIL: storeOutput names an undeclared output");
    auto value = call->getArgOperand(4);
    switch (e->kind) {
    case SemanticKind::Depth:
    case SemanticKind::DepthLessEqual:
    case SemanticKind::DepthGreaterEqual:
      if (res.depth_output_reg)
        ir.CreateStore(Cast(value, ir.getFloatTy()), res.depth_output_reg);
      break;
    case SemanticKind::Coverage:
      ir.CreateStore(Cast(value, ir.getInt32Ty()), res.coverage_mask_reg);
      break;
    case SemanticKind::StencilRef:
      ir.CreateStore(Cast(value, ir.getInt32Ty()), res.stencil_ref_reg);
      break;
    default: {
      auto word = value->getType()->isIntegerTy(1) ? ir.CreateSExt(value, ir.getInt32Ty()) : Cast(value, ir.getInt32Ty());
      ir.CreateStore(word, RegisterElementPointer(res.output, *e, call->getArgOperand(2), ConstantU32(call->getArgOperand(3))));
    }
    }
    Replace(call, nullptr);
    return llvm::Error::success();
  }
  case op::Discard: { // (82, i1 condition)
    auto cond = call->getArgOperand(1);
    if (auto c = llvm::dyn_cast<llvm::ConstantInt>(cond); c && c->isZero()) {
      Replace(call, nullptr);
      return llvm::Error::success();
    }
    auto then = llvm::SplitBlockAndInsertIfThen(cond, call, false);
    ir.SetInsertPoint(then);
    air.CreateDiscard();
    Replace(call, nullptr);
    return llvm::Error::success();
  }
  case op::DerivCoarseX:
  case op::DerivCoarseY:
  case op::DerivFineX:
  case op::DerivFineY:
    Replace(call, air.CreateDerivative(call->getArgOperand(1), opcode == op::DerivCoarseY || opcode == op::DerivFineY));
    return llvm::Error::success();
  case op::SampleIndex:
  case op::Coverage:
  case op::PrimitiveID: {
    auto kind = opcode == op::SampleIndex ? SemanticKind::SampleIndex
                : opcode == op::Coverage  ? SemanticKind::Coverage
                                          : SemanticKind::PrimitiveID;
    auto v = UnpackedInputOfKind(kind);
    if (!v)
      return Unsupported(call);
    Replace(call, Cast(v, call->getType()));
    return llvm::Error::success();
  }
  case op::WaveIsFirstLane:
    Replace(call, Simd(ir, "air.simd_is_first", ir.getInt1Ty(), {}));
    return llvm::Error::success();
  case op::WaveGetLaneIndex:
  case op::WaveGetLaneCount: {
    uint32_t arg = opcode == op::WaveGetLaneIndex ? dxil.lane_index_arg : dxil.lane_count_arg;
    Replace(call, ctx.function->getArg(arg));
    return llvm::Error::success();
  }
  case op::WaveAnyTrue:
  case op::WaveAllTrue:
    Replace(call, Simd(ir, opcode == op::WaveAnyTrue ? "air.simd_any" : "air.simd_all", ir.getInt1Ty(), {call->getArgOperand(1)}));
    return llvm::Error::success();
  case op::WaveActiveAllEqual: { // (115, T)
    auto v = call->getArgOperand(1);
    auto first = Simd(ir, "air.simd_broadcast_first" + SimdSuffix(v->getType(), false), v->getType(), {v});
    auto same = v->getType()->isFloatingPointTy() ? ir.CreateFCmpOEQ(v, first) : ir.CreateICmpEQ(v, first);
    Replace(call, Simd(ir, "air.simd_all", ir.getInt1Ty(), {same}));
    return llvm::Error::success();
  }
  case op::WaveActiveBallot: { // (116, i1) -> %dx.types.fouri32 {lanes 0-31, lanes 32-63, 0, 0}
    auto ballot = Simd(ir, "air.simd_ballot.i64", ir.getInt64Ty(), {call->getArgOperand(1)});
    Replace(call, Aggregate(call->getType(), {ir.CreateTrunc(ballot, ir.getInt32Ty()),
                                              ir.CreateTrunc(ir.CreateLShr(ballot, 32), ir.getInt32Ty()),
                                              ir.getInt32(0), ir.getInt32(0)}));
    return llvm::Error::success();
  }
  case op::QuadReadLaneAt: { // (122, T, i32 quadLane)
    auto v = call->getArgOperand(1);
    auto lane = ir.CreateTrunc(call->getArgOperand(2), ir.getInt16Ty());
    Replace(call, Simd(ir, "air.quad_shuffle" + SimdSuffix(v->getType(), false), v->getType(), {v, lane}));
    return llvm::Error::success();
  }
  case op::QuadOp: { // (123, T, i8 op): read across X (0), Y (1) or the diagonal (2): the quad lane XOR 1, 2 or 3
    auto v = call->getArgOperand(1);
    auto mask = ir.getInt16(ConstantU32(call->getArgOperand(2)) + 1);
    Replace(call, Simd(ir, "air.quad_shuffle_xor" + SimdSuffix(v->getType(), false), v->getType(), {v, mask}));
    return llvm::Error::success();
  }
  case op::WaveReadLaneAt: { // (117, T, i32 lane)
    auto v = call->getArgOperand(1);
    auto lane = ir.CreateTrunc(call->getArgOperand(2), ir.getInt16Ty());
    Replace(call, Simd(ir, "air.simd_shuffle" + SimdSuffix(v->getType(), false), v->getType(), {v, lane}));
    return llvm::Error::success();
  }
  case op::WaveReadLaneFirst: { // (118, T)
    auto v = call->getArgOperand(1);
    Replace(call, Simd(ir, "air.simd_broadcast_first" + SimdSuffix(v->getType(), false), v->getType(), {v}));
    return llvm::Error::success();
  }
  case op::WaveActiveOp:
  case op::WavePrefixOp: { // (119/121, T, i8 op: 0 sum, 1 product, 2 min, 3 max, i8 sign: 0 signed, 1 unsigned)
    auto v = call->getArgOperand(1);
    uint32_t kind = ConstantU32(call->getArgOperand(2));
    bool is_signed = ConstantU32(call->getArgOperand(3)) == 0;
    static const char *const active[] = {"air.simd_sum", "air.simd_product", "air.simd_min", "air.simd_max"};
    static const char *const prefix[] = {"air.simd_prefix_exclusive_sum", "air.simd_prefix_exclusive_product"};
    if (kind > 3 || (opcode == op::WavePrefixOp && kind > 1))
      return Unsupported(call);
    auto name = std::string(opcode == op::WaveActiveOp ? active[kind] : prefix[kind]) + SimdSuffix(v->getType(), is_signed);
    Replace(call, Simd(ir, name, v->getType(), {v}));
    return llvm::Error::success();
  }
  case op::WaveActiveBit: { // (120, T, i8 op: 0 and, 1 or, 2 xor)
    auto v = call->getArgOperand(1);
    static const char *const bit[] = {"air.simd_and", "air.simd_or", "air.simd_xor"};
    uint32_t kind = ConstantU32(call->getArgOperand(2));
    if (kind > 2)
      return Unsupported(call);
    Replace(call, Simd(ir, std::string(bit[kind]) + SimdSuffix(v->getType(), false), v->getType(), {v}));
    return llvm::Error::success();
  }
  case op::WaveAllBitCount:
  case op::WavePrefixBitCount: { // (135/136, i1): lanes (below this one) where the value is true
    llvm::Value *ballot = Simd(ir, "air.simd_ballot.i64", ir.getInt64Ty(), {call->getArgOperand(1)});
    if (opcode == op::WavePrefixBitCount) {
      auto lane = ir.CreateZExt(ctx.function->getArg(dxil.lane_index_arg), ir.getInt64Ty());
      ballot = ir.CreateAnd(ballot, ir.CreateSub(ir.CreateShl(ir.getInt64(1), lane), ir.getInt64(1)));
    }
    auto lo = air.CreateIntUnOp(llvm::air::AIRBuilder::popcount, ir.CreateTrunc(ballot, ir.getInt32Ty()));
    auto hi = air.CreateIntUnOp(llvm::air::AIRBuilder::popcount, ir.CreateTrunc(ir.CreateLShr(ballot, 32), ir.getInt32Ty()));
    Replace(call, ir.CreateAdd(lo, hi));
    return llvm::Error::success();
  }
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
