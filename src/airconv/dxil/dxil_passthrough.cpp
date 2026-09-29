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
#include "dxil.hpp"
#include "../airconv_context.hpp"
#include "../dxbc_converter.hpp"
#include "llvm/IR/IRBuilder.h"

namespace dxmt::dxil {

using namespace dxmt::dxbc;

llvm::Expected<std::unique_ptr<llvm::Module>> BuildPassThroughVertex(const EntryInfo &pixel, llvm::LLVMContext &context) {
  auto module = std::make_unique<llvm::Module>("shader.air", context);
  initializeModule(*module);
  setup_metal_version(*module, SM50_SHADER_METAL_310);
  air::FunctionSignatureBuilder fs;
  fs.DefineOutput(air::OutputPosition{.type = air::msl_float4});
  for (auto &e : pixel.inputs)
    if (e.kind == SemanticKind::Arbitrary)
      for (uint32_t r = 0; r < e.rows; r++)
        fs.DefineOutput(Varying(e, r));
  auto [function, metadata] = fs.CreateFunction("vs_passthrough", context, *module, 0, false);
  llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", function));
  auto zero = llvm::ConstantAggregateZero::get(function->getReturnType());
  auto f = [&](float v) { return llvm::ConstantFP::get(builder.getFloatTy(), v); };
  builder.CreateRet(builder.CreateInsertValue(zero, llvm::ConstantVector::get({f(0), f(0), f(0), f(1)}), {0}));
  module->getOrInsertNamedMetadata("air.vertex")->addOperand(metadata);
  return std::move(module);
}

} // namespace dxmt::dxil
