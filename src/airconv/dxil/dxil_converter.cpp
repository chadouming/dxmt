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
#include "../airconv_context.hpp"
#include "../airconv_error.hpp"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

namespace dxmt::dxil {

using namespace dxmt::dxbc;

namespace {

// Removes what only DXIL tools read: named metadata, instruction metadata, and DXIL's type names.
void StripDXIL(llvm::Module &module) {
  llvm::SmallVector<llvm::NamedMDNode *, 16> named;
  for (auto &n : module.named_metadata())
    if (n.getName().startswith("dx.") || n.getName() == "llvm.ident")
      named.push_back(&n);
  for (auto *n : named)
    module.eraseNamedMetadata(n);
  for (auto &fn : module)
    for (auto &bb : fn)
      for (auto &inst : bb)
        inst.dropUnknownNonDebugMetadata();
  for (auto *st : module.getIdentifiedStructTypes())
    if (st->getName().startswith("dx.types."))
      st->setName("dxil." + st->getName().substr(9).str());
}

} // namespace

llvm::Expected<std::unique_ptr<llvm::Module>>
ConvertDXIL(SM50ShaderInternal *shader, const char *name, llvm::LLVMContext &context, SM50_SHADER_COMPILATION_ARGUMENT_DATA *pArgs) {
  auto &dxil = *shader->dxil;
  auto &entry = dxil.entry;
  auto loaded = LoadModule(context, dxil.bitcode.data(), dxil.bitcode.size());
  if (!loaded)
    return loaded.takeError();
  std::unique_ptr<llvm::Module> module = std::move(*loaded);
  llvm::Function *dxil_main = module->getFunction(entry.name);
  if (!dxil_main)
    return llvm::make_error<UnsupportedFeature>("DXIL: entry function missing");
  initializeModule(*module); // AIR triple, data layout, SDK version, module flags

  // From here on, as convert_dxbc_compute_shader (dxbc_converter.cpp), except that the body comes from DXIL.
  auto func_signature = shader->func_signature; // copy
  auto shader_info = &shader->shader_info;
  SM50_SHADER_METAL_VERSION metal_version = SM50_SHADER_METAL_310;
  SM50_SHADER_FLAG shader_flags = {};
  SM50_SHADER_COMMON_DATA *sm50_common = nullptr;
  if (args_get_data<SM50_SHADER_COMMON, SM50_SHADER_COMMON_DATA>(pArgs, &sm50_common)) {
    metal_version = sm50_common->metal_version;
    shader_flags = sm50_common->flags;
  }
  SM50_SHADER_ROOT_SIGNATURE_DATA *rootsig = nullptr;
  args_get_data<SM50_SHADER_ROOT_SIGNATURE, SM50_SHADER_ROOT_SIGNATURE_DATA>(pArgs, &rootsig);

  IREffect prologue([](auto) { return std::monostate(); });
  IRValue epilogue([](struct context ctx) -> pvalue {
    auto retTy = ctx.function->getReturnType();
    if (retTy->isVoidTy())
      return nullptr;
    return llvm::ConstantAggregateZero::get(retTy);
  });
  io_binding_map resource_map;
  air::AirType types(context);
  {
    SignatureContext sig_ctx(prologue, epilogue, func_signature, resource_map);
    for (auto &p : shader->signature_handlers)
      p(sig_ctx);
  }
  auto binding_map = rootsig ? setup_binding_rootsig(shader_info, func_signature, *module, shader->shader_type,
                                                     rootsig->bytecode, rootsig->bytecode_length)
                             : setup_binding_table2(shader_info, func_signature, *module);

  auto [function, function_metadata] = func_signature.CreateFunction(name, context, *module, 0, false);
  auto entry_bb = llvm::BasicBlock::Create(context, "entry", function);
  auto epilogue_bb = llvm::BasicBlock::Create(context, "epilogue", function);
  llvm::IRBuilder<> builder(entry_bb);
  llvm::raw_null_ostream nulldbg{};
  llvm::air::AIRBuilder air(
      {
          .sampleNaNToZero = bool(shader_flags & SM50_SHADER_FLAG_SAMPLE_NAN_TO_ZERO),
          .defuseFma = bool(shader_flags & SM50_SHADER_FLAG_DEFUSE_FMA),
      },
      builder, nulldbg
  );
  setup_metal_version(*module, metal_version);

  struct context ctx{
      .builder = builder,
      .air = air,
      .binding = *binding_map,
      .llvm = context,
      .module = *module,
      .function = function,
      .resource = resource_map,
      .types = types,
      .pso_sample_mask = 0xffffffff,
      .shader_type = shader->shader_type,
      .metal_version = metal_version,
  };
  if (auto err = prologue.build(ctx).takeError())
    return std::move(err);

  // The DXIL body: move its blocks after `epilogue`, enter them from `entry`, and leave through `epilogue`.
  function->getBasicBlockList().splice(function->end(), dxil_main->getBasicBlockList());
  builder.CreateBr(epilogue_bb->getNextNode());
  for (auto &bb : *function)
    if (auto ret = llvm::dyn_cast_or_null<llvm::ReturnInst>(bb.getTerminator())) {
      llvm::BranchInst::Create(epilogue_bb, ret);
      ret->eraseFromParent();
    }

  // Lower every dx.op call (collected first: lowering erases them).
  std::vector<llvm::CallInst *> calls;
  for (auto &bb : *function)
    for (auto &inst : bb)
      if (auto call = llvm::dyn_cast<llvm::CallInst>(&inst); call && OpCode(*call) != ~0u)
        calls.push_back(call);
  Lowering lowering(entry, ctx);
  for (auto *call : calls)
    if (auto err = lowering.Lower(call))
      return std::move(err);
  lowering.Cleanup(*function);

  builder.SetInsertPoint(epilogue_bb);
  if (auto err = epilogue.build(ctx).takeError())
    return std::move(err);
  builder.CreateRetVoid();

  dxil_main->eraseFromParent();
  for (auto it = module->begin(); it != module->end();) {
    auto &fn = *it++;
    if (fn.getName().startswith("dx.op.") && fn.use_empty())
      fn.eraseFromParent();
  }
  StripDXIL(*module);
  module->getOrInsertNamedMetadata("air.kernel")->addOperand(function_metadata);

  std::string problems;
  llvm::raw_string_ostream os(problems);
  if (llvm::verifyModule(*module, &os))
    return llvm::make_error<UnsupportedFeature>("DXIL: invalid module after lowering: " + os.str());
  return std::move(module);
}

} // namespace dxmt::dxil
