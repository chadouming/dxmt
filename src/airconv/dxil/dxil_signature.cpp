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
#include "../airconv_error.hpp"
#include "../dxbc_converter.hpp"

namespace dxmt::dxil {

using namespace dxmt::dxbc;
using namespace dxmt::air;

namespace {

// Defines a function input and, in the prologue, stores the argument into one of io_binding_map's special registers
// (as handle_signature_cs does in dxbc_signature.cpp).
template <typename Input>
void DefineSpecial(SM50ShaderInternal *shader, llvm::Value *io_binding_map::*field) {
  auto index = shader->func_signature.DefineInput(Input{});
  shader->signature_handlers.push_back([=](SignatureContext &sig) {
    sig.prologue << make_effect([=](struct context ctx) {
      ctx.resource.*field = ctx.function->getArg(index);
      return std::monostate{};
    });
  });
}

llvm::Error
AddComputeSignatureHandlers(const llvm::Module &module, SM50ShaderInternal *shader) {
  // Compute IDs aren't signature elements in DXIL: define the inputs the module's dx.op calls read.
  if (ModuleCallsOp(module, op::ThreadId))
    DefineSpecial<InputThreadPositionInGrid>(shader, &io_binding_map::thread_id_arg);
  if (ModuleCallsOp(module, op::GroupId))
    DefineSpecial<InputThreadgroupPositionInGrid>(shader, &io_binding_map::thread_group_id_arg);
  if (ModuleCallsOp(module, op::ThreadIdInGroup))
    DefineSpecial<InputThreadPositionInThreadgroup>(shader, &io_binding_map::thread_id_in_group_arg);
  if (ModuleCallsOp(module, op::FlattenedThreadIdInGroup))
    DefineSpecial<InputThreadIndexInThreadgroup>(shader, &io_binding_map::thread_id_in_group_flat_arg);
  return llvm::Error::success();
}

} // namespace

llvm::Error
AddSignatureHandlers(const EntryInfo &entry, const llvm::Module &module, SM50ShaderInternal *shader) {
  if (entry.kind == ShaderKind::Compute)
    return AddComputeSignatureHandlers(module, shader);
  return llvm::make_error<UnsupportedFeature>("DXIL: vertex and pixel shaders not supported yet");
}

} // namespace dxmt::dxil
