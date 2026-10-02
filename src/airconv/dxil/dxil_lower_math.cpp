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
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include <cfloat>

namespace dxmt::dxil {

using FP = llvm::air::AIRBuilder;

llvm::Error
Lowering::LowerMath(uint32_t opcode, llvm::CallInst *call) {
  auto arg = [&](unsigned i) { return call->getArgOperand(i); };
  auto ty = call->getType();
  auto done = [&](llvm::Value *v) {
    Replace(call, v);
    return llvm::Error::success();
  };
  // Metal's fast variants, as airconv's DXBC path uses them.
  auto un = [&](FP::FPUnOp fn) { return done(air.CreateFPUnOp(fn, arg(1))); };

  switch (opcode) {
  case op::FAbs: return un(FP::fabs);
  case op::Saturate: return un(FP::saturate);
  case op::Cos: return un(FP::cos);
  case op::Sin: return un(FP::sin);
  case op::Tan: return un(FP::tan);
  case op::Acos: return un(FP::acos);
  case op::Asin: return un(FP::asin);
  case op::Atan: return un(FP::atan);
  case op::Hcos: return un(FP::cosh);
  case op::Hsin: return un(FP::sinh);
  case op::Htan: return un(FP::tanh);
  case op::Exp: return un(FP::exp2); // DXIL's Exp and Log are base 2
  case op::Log: return un(FP::log2);
  case op::Frc: return un(FP::fract);
  case op::Sqrt: return un(FP::sqrt);
  case op::Rsqrt: return un(FP::rsqrt);
  case op::Round_ne: return un(FP::rint);
  case op::Round_ni: return un(FP::floor);
  case op::Round_pi: return un(FP::ceil);
  case op::Round_z: return un(FP::trunc);

  case op::IsNaN: return done(air.CreateIsNaN(arg(1)));
  case op::IsInf:
  case op::IsFinite:
  case op::IsNormal: {
    auto x = air.CreateFPUnOp(FP::fabs, arg(1), false);
    auto inf = llvm::ConstantFP::getInfinity(x->getType());
    if (opcode == op::IsInf)
      return done(ir.CreateFCmpOEQ(x, inf));
    if (opcode == op::IsFinite)
      return done(ir.CreateFCmpOLT(x, inf));
    auto min_normal = llvm::ConstantFP::get(x->getType(), x->getType()->isHalfTy() ? 6.103515625e-05 : FLT_MIN);
    return done(ir.CreateAnd(ir.CreateFCmpOGE(x, min_normal), ir.CreateFCmpOLT(x, inf)));
  }

  case op::Bfrev: return done(air.CreateIntUnOp(FP::reverse_bits, arg(1)));
  case op::Countbits: return done(ir.CreateZExtOrTrunc(air.CreateIntUnOp(FP::popcount, arg(1)), ty));
  case op::FirstbitLo:
  case op::FirstbitHi:
  case op::FirstbitSHi: {
    // FirstbitLo counts from the least significant bit; FirstbitHi/SHi count from the most significant one (DXC turns
    // them into HLSL's firstbithigh itself). ~0u when no bit qualifies.
    auto x = arg(1);
    if (opcode == op::FirstbitSHi) // first bit that differs from the sign
      x = ir.CreateSelect(ir.CreateICmpSLT(x, llvm::ConstantInt::get(x->getType(), 0)), ir.CreateNot(x), x);
    auto zeros = ir.CreateZExtOrTrunc(air.CreateCountZero(x, opcode == op::FirstbitLo), ty);
    return done(ir.CreateSelect(ir.CreateICmpEQ(x, llvm::ConstantInt::get(x->getType(), 0)), llvm::ConstantInt::get(ty, ~0u), zeros));
  }

  case op::FMax: return done(air.CreateFPBinOp(FP::fmax, arg(1), arg(2)));
  case op::FMin: return done(air.CreateFPBinOp(FP::fmin, arg(1), arg(2)));
  case op::IMax: return done(air.CreateIntBinOp(FP::max, arg(1), arg(2), true));
  case op::IMin: return done(air.CreateIntBinOp(FP::min, arg(1), arg(2), true));
  case op::UMax: return done(air.CreateIntBinOp(FP::max, arg(1), arg(2), false));
  case op::UMin: return done(air.CreateIntBinOp(FP::min, arg(1), arg(2), false));

  case op::IMul:
  case op::UMul: { // -> {hi, lo}
    bool s = opcode == op::IMul;
    auto wide = ir.getInt64Ty();
    auto a = s ? ir.CreateSExt(arg(1), wide) : ir.CreateZExt(arg(1), wide);
    auto b = s ? ir.CreateSExt(arg(2), wide) : ir.CreateZExt(arg(2), wide);
    auto p = ir.CreateMul(a, b);
    return done(Aggregate(ty, {ir.CreateTrunc(ir.CreateLShr(p, 32), ir.getInt32Ty()), ir.CreateTrunc(p, ir.getInt32Ty())}));
  }
  case op::UDiv: { // -> {quotient, remainder}; D3D: division by zero gives ~0u for both
    auto zero = ir.CreateICmpEQ(arg(2), ir.getInt32(0));
    auto safe = ir.CreateSelect(zero, ir.getInt32(1), arg(2));
    auto all = ir.getInt32(~0u);
    return done(Aggregate(ty, {ir.CreateSelect(zero, all, ir.CreateUDiv(arg(1), safe)),
                               ir.CreateSelect(zero, all, ir.CreateURem(arg(1), safe))}));
  }
  case op::UAddc:
  case op::USubb: { // -> {result, carry/borrow}
    auto id = opcode == op::UAddc ? llvm::Intrinsic::uadd_with_overflow : llvm::Intrinsic::usub_with_overflow;
    auto r = ir.CreateBinaryIntrinsic(id, arg(1), arg(2));
    return done(Aggregate(ty, {ir.CreateExtractValue(r, {0}),
                               ir.CreateZExtOrTrunc(ir.CreateExtractValue(r, {1}), ty->getStructElementType(1))}));
  }

  case op::FMad: return done(ir.CreateFAdd(ir.CreateFMul(arg(1), arg(2)), arg(3))); // not fused, like D3D's mad
  case op::Fma: return done(air.CreateFMA(arg(1), arg(2), arg(3)));
  case op::IMad:
  case op::UMad: return done(ir.CreateAdd(ir.CreateMul(arg(1), arg(2)), arg(3)));
  case op::Ibfe:
  case op::Ubfe: { // (width, offset, value), as D3D's ibfe/ubfe
    auto w = ir.CreateAnd(arg(1), 31), o = ir.CreateAnd(arg(2), 31), v = arg(3);
    auto fits = ir.CreateICmpULT(ir.CreateAdd(w, o), ir.getInt32(32));
    auto shifted = ir.CreateShl(v, ir.CreateSub(ir.getInt32(32), ir.CreateAdd(w, o)));
    auto extracted = opcode == op::Ibfe ? ir.CreateAShr(shifted, ir.CreateSub(ir.getInt32(32), w))
                                        : ir.CreateLShr(shifted, ir.CreateSub(ir.getInt32(32), w));
    auto tail = opcode == op::Ibfe ? ir.CreateAShr(v, o) : ir.CreateLShr(v, o);
    auto r = ir.CreateSelect(fits, extracted, tail);
    return done(ir.CreateSelect(ir.CreateICmpEQ(w, ir.getInt32(0)), ir.getInt32(0), r));
  }
  case op::Bfi: { // (width, offset, insert, base)
    auto w = ir.CreateAnd(arg(1), 31), o = ir.CreateAnd(arg(2), 31);
    auto mask = ir.CreateShl(ir.CreateSub(ir.CreateShl(ir.getInt32(1), w), ir.getInt32(1)), o);
    return done(ir.CreateOr(ir.CreateAnd(arg(4), ir.CreateNot(mask)), ir.CreateAnd(ir.CreateShl(arg(3), o), mask)));
  }

  case op::Dot2:
  case op::Dot3:
  case op::Dot4: {
    unsigned n = opcode - op::Dot2 + 2;
    auto vec_ty = llvm::FixedVectorType::get(ty, n);
    llvm::Value *a = llvm::UndefValue::get(vec_ty), *b = a;
    for (unsigned i = 0; i < n; i++) {
      a = ir.CreateInsertElement(a, arg(1 + i), i);
      b = ir.CreateInsertElement(b, arg(1 + n + i), i);
    }
    return done(air.CreateDotProduct(a, b));
  }

  case op::MakeDouble: { // (lo, hi)
    auto v = ir.CreateInsertElement(ir.CreateInsertElement(llvm::UndefValue::get(air.getIntTy(2)), arg(1), (uint64_t)0), arg(2), 1);
    return done(ir.CreateBitCast(v, ty));
  }
  case op::SplitDouble: { // -> {lo, hi}
    auto v = ir.CreateBitCast(arg(1), air.getIntTy(2));
    return done(Aggregate(ty, {ir.CreateExtractElement(v, (uint64_t)0), ir.CreateExtractElement(v, 1)}));
  }
  case op::BitcastI16toF16:
  case op::BitcastF16toI16:
  case op::BitcastI32toF32:
  case op::BitcastF32toI32:
    return done(ir.CreateBitCast(arg(1), ty));
  case op::LegacyF32ToF16: { // float -> low 16 bits of an i32, rounded toward zero (as D3DMetal)
    auto nearest = ir.CreateFPTrunc(arg(1), ir.getHalfTy());
    auto grew = ir.CreateFCmpOGT(air.CreateFPUnOp(FP::fabs, ir.CreateFPExt(nearest, ir.getFloatTy()), false),
                                 air.CreateFPUnOp(FP::fabs, arg(1), false));
    auto bits = ir.CreateBitCast(nearest, ir.getInt16Ty());
    auto rtz = ir.CreateSelect(grew, ir.CreateSub(bits, ir.getInt16(1)), bits); // one unit toward zero
    return done(ir.CreateZExt(rtz, ty));
  }
  case op::LegacyF16ToF32:
    return done(ir.CreateFPExt(ir.CreateBitCast(ir.CreateTrunc(arg(1), ir.getInt16Ty()), ir.getHalfTy()), ty));

  case op::Dot2AddHalf: { // (acc, ax, ay, bx, by): acc + a.x*b.x + a.y*b.y, in float
    auto f = [&](unsigned i) { return ir.CreateFPExt(arg(i), ir.getFloatTy()); };
    return done(ir.CreateFAdd(arg(1), ir.CreateFAdd(ir.CreateFMul(f(2), f(4)), ir.CreateFMul(f(3), f(5)))));
  }
  case op::Dot4AddI8Packed:
  case op::Dot4AddU8Packed: { // (acc, a, b): acc + sum of the byte products
    bool s = opcode == op::Dot4AddI8Packed;
    llvm::Value *sum = arg(1);
    for (unsigned k = 0; k < 4; k++) {
      auto byte = [&](llvm::Value *x) {
        auto t = ir.CreateTrunc(ir.CreateLShr(x, 8 * k), ir.getInt8Ty());
        return s ? ir.CreateSExt(t, ir.getInt32Ty()) : ir.CreateZExt(t, ir.getInt32Ty());
      };
      sum = ir.CreateAdd(sum, ir.CreateMul(byte(arg(2)), byte(arg(3))));
    }
    return done(sum);
  }
  case op::Unpack4x8: { // (mode: 0 unsigned, 1 signed, packed) -> four bytes, extended to i16 or i32
    bool s = llvm::cast<llvm::ConstantInt>(arg(1))->getZExtValue() == 1;
    auto elem = ty->getStructElementType(0);
    llvm::SmallVector<llvm::Value *, 4> bytes;
    for (unsigned k = 0; k < 4; k++) {
      auto t = ir.CreateTrunc(ir.CreateLShr(arg(2), 8 * k), ir.getInt8Ty());
      bytes.push_back(s ? ir.CreateSExt(t, elem) : ir.CreateZExt(t, elem));
    }
    return done(Aggregate(ty, bytes));
  }
  case op::Pack4x8: { // (mode: 0 truncate, 1 clamp to [0, 255], 2 clamp to [-128, 127], x, y, z, w) -> i32
    auto mode = llvm::cast<llvm::ConstantInt>(arg(1))->getZExtValue();
    llvm::Value *packed = ir.getInt32(0);
    for (unsigned k = 0; k < 4; k++) {
      llvm::Value *v = arg(2 + k);
      if (mode != 0) {
        auto lo = llvm::ConstantInt::get(v->getType(), mode == 1 ? 0 : -128, true);
        auto hi = llvm::ConstantInt::get(v->getType(), mode == 1 ? 255 : 127, true);
        v = ir.CreateSelect(ir.CreateICmpSLT(v, lo), lo, v);
        v = ir.CreateSelect(ir.CreateICmpSGT(v, hi), hi, v);
      }
      auto byte = ir.CreateZExt(ir.CreateTrunc(v, ir.getInt8Ty()), ir.getInt32Ty());
      packed = ir.CreateOr(packed, ir.CreateShl(byte, 8 * k));
    }
    return done(packed);
  }
  default:
    return Unsupported(call);
  }
}

} // namespace dxmt::dxil
