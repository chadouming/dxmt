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
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/ModuleSlotTracker.h"
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

// Points every use of `from` (a pointer) at `to`, the same object in another address space: GEPs, bitcasts and loads
// are rebuilt or repointed; constant expressions become instructions first.
llvm::Error Retarget(llvm::Value *from, llvm::Value *to) {
  for (auto *user : llvm::make_early_inc_range(from->users())) {
    if (auto ce = llvm::dyn_cast<llvm::ConstantExpr>(user)) {
      for (auto *ce_user : llvm::make_early_inc_range(ce->users())) {
        auto inst = llvm::dyn_cast<llvm::Instruction>(ce_user);
        if (!inst)
          return llvm::make_error<UnsupportedFeature>("DXIL: a static const array in a constant initializer");
        auto expanded = ce->getAsInstruction(inst);
        inst->replaceUsesOfWith(ce, expanded);
      }
      ce->destroyConstant();
    }
  }
  for (auto *user : llvm::make_early_inc_range(from->users())) {
    unsigned space = to->getType()->getPointerAddressSpace();
    if (auto gep = llvm::dyn_cast<llvm::GetElementPtrInst>(user)) {
      llvm::SmallVector<llvm::Value *, 4> indices(gep->indices());
      auto moved = llvm::GetElementPtrInst::Create(gep->getSourceElementType(), to, indices, gep->getName(), gep);
      moved->setIsInBounds(gep->isInBounds());
      if (auto err = Retarget(gep, moved))
        return err;
      gep->eraseFromParent();
    } else if (auto cast = llvm::dyn_cast<llvm::BitCastInst>(user)) {
      auto elem = cast->getDestTy()->getNonOpaquePointerElementType();
      auto moved = new llvm::BitCastInst(to, elem->getPointerTo(space), cast->getName(), cast);
      if (auto err = Retarget(cast, moved))
        return err;
      cast->eraseFromParent();
    } else if (auto load = llvm::dyn_cast<llvm::LoadInst>(user)) {
      load->setOperand(load->getPointerOperandIndex(), to);
    } else {
      return llvm::make_error<UnsupportedFeature>("DXIL: a static const array used other than by loads");
    }
  }
  return llvm::Error::success();
}

// DXIL keeps dynamically indexed `static const` arrays as address space 0 globals, which Metal can't link ("Undefined
// symbols"); they belong in constant space, as airconv's immediate constant buffers. Mutable statics aren't supported.
llvm::Error MoveConstantsToConstantSpace(llvm::Module &module) {
  for (auto &gv : llvm::make_early_inc_range(module.globals())) {
    if (gv.getAddressSpace() != 0 || gv.getName().startswith("llvm."))
      continue;
    if (!gv.isConstant() || !gv.hasInitializer())
      return llvm::make_error<UnsupportedFeature>("DXIL: static (non-const) global " + gv.getName().str() + " not supported");
    auto moved = new llvm::GlobalVariable(module, gv.getValueType(), true, gv.getLinkage(), gv.getInitializer(), "",
                                          nullptr, llvm::GlobalValue::NotThreadLocal, 2);
    moved->takeName(&gv);
    moved->setAlignment(gv.getAlign());
    if (auto err = Retarget(&gv, moved))
      return err;
    gv.eraseFromParent();
  }
  return llvm::Error::success();
}

// The shader's module, set up for AIR, and its DXIL entry function (renamed: the caller may ask for its name, "main").
llvm::Expected<std::pair<std::unique_ptr<llvm::Module>, llvm::Function *>>
Load(const DXILShader &dxil, llvm::LLVMContext &context) {
  auto loaded = LoadModule(context, dxil.bitcode.data(), dxil.bitcode.size());
  if (!loaded)
    return loaded.takeError();
  std::unique_ptr<llvm::Module> module = std::move(*loaded);
  llvm::Function *dxil_main = module->getFunction(dxil.entry.name);
  if (!dxil_main)
    return llvm::make_error<UnsupportedFeature>("DXIL: entry function missing");
  dxil_main->setName("dxil.entry");
  initializeModule(*module); // AIR triple, data layout, SDK version, module flags
  return std::make_pair(std::move(module), dxil_main);
}

// Shader probe (MacNeutron, DXMT_PROBE=<capture hash>:<a>,<b>,<c>): the pixel shader capture mode saves as
// ps-<hash>.dxil outputs three of its own DXIL values, numbered as `dxc -dumpbin` prints them (%a, %b, %c), as
// SV_Target0's red, green and blue, alpha 1: a frame dump then shows a shader's intermediate values. A value defined on
// a path the pixel didn't take reads -12345; NaN reads -7777 and infinities +-8888. Scalars only: probe a sample's or a
// cbuffer load's extractvalue.
void
ApplyProbe(const DXILShader &dxil, llvm::Function &main) {
  static const std::string spec = [] {
    const char *value = getenv("DXMT_PROBE");
    return std::string(value ? value : "");
  }();
  // "<hash>:<a>,<b>,<c>[,a]": a channel written "-" keeps the shader's own value; ",a" keeps its own alpha too.
  unsigned long long hash = 0;
  unsigned numbers[3] = {~0u, ~0u, ~0u};
  bool keep_alpha = false;
  auto colon = spec.find(':');
  if (dxil.entry.kind != ShaderKind::Pixel || colon == std::string::npos ||
      sscanf(spec.c_str(), "%llx", &hash) != 1 || hash != dxil.capture_hash)
    return;
  {
    size_t at = colon + 1;
    for (int k = 0; k < 4 && at <= spec.size(); k++) {
      size_t comma = spec.find(',', at);
      std::string field = spec.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
      if (k == 3)
        keep_alpha = field == "a";
      else if (field != "-")
        numbers[k] = (unsigned)strtoul(field.c_str(), nullptr, 10);
      if (comma == std::string::npos)
        break;
      at = comma + 1;
    }
  }
  llvm::ModuleSlotTracker slots(main.getParent());
  slots.incorporateFunction(main);
  llvm::Instruction *probed[3] = {};
  for (auto &block : main)
    for (auto &inst : block)
      for (int k = 0; k < 3; k++)
        if (slots.getLocalSlot(&inst) == (int)numbers[k])
          probed[k] = &inst;
  llvm::IRBuilder<> entry(&*main.getEntryBlock().getFirstInsertionPt());
  auto f32 = entry.getFloatTy();
  llvm::Value *slots_out[3];
  for (int k = 0; k < 3; k++) {
    slots_out[k] = entry.CreateAlloca(f32);
    entry.CreateStore(llvm::ConstantFP::get(f32, -12345.0), slots_out[k]);
  }
  for (int k = 0; k < 3; k++) {
    auto *inst = probed[k];
    std::string text = "missing";
    if (inst) {
      llvm::raw_string_ostream os(text);
      text.clear();
      inst->print(os, slots);
      os.flush();
      llvm::IRBuilder<> at(inst->getParent(), llvm::isa<llvm::PHINode>(inst)
                                                  ? inst->getParent()->getFirstInsertionPt()
                                                  : std::next(llvm::BasicBlock::iterator(inst)));
      llvm::Value *value = inst;
      auto type = value->getType();
      if (type->isIntegerTy())
        value = at.CreateUIToFP(value, f32);
      else if (type->isHalfTy() || type->isDoubleTy())
        value = at.CreateFPCast(value, f32);
      if (value->getType()->isFloatTy()) {
        // NaN and infinities as markers: blending would turn them into 0 on the way to the dump.
        auto inf = llvm::ConstantFP::getInfinity(f32);
        auto marked = at.CreateSelect(at.CreateFCmpUNO(value, value), llvm::ConstantFP::get(f32, -7777.0), value);
        marked = at.CreateSelect(at.CreateFCmpOEQ(value, inf), llvm::ConstantFP::get(f32, 8888.0), marked);
        marked = at.CreateSelect(at.CreateFCmpOEQ(value, llvm::ConstantFP::getInfinity(f32, true)),
                                 llvm::ConstantFP::get(f32, -8888.0), marked);
        at.CreateStore(marked, slots_out[k]);
      }
      else
        text = "not a scalar: " + text;
    }
    if (numbers[k] != ~0u)
      fprintf(stderr, "DXMT_PROBE ps-%016llx channel %d: %%%u = %s\n", hash, k, numbers[k], text.substr(0, 160).c_str());
  }
  uint32_t target = ~0u;
  for (auto &e : dxil.entry.outputs)
    if (e.kind == SemanticKind::Target && (e.semantic_indices.empty() || e.semantic_indices[0] == 0))
      target = e.id;
  for (auto &block : main)
    for (auto &inst : block)
      if (auto call = llvm::dyn_cast<llvm::CallInst>(&inst);
          call && OpCode(*call) == op::StoreOutput && ConstantU32(call->getArgOperand(1), ~0u) == target &&
          call->getArgOperand(4)->getType()->isFloatTy()) {
        uint32_t col = ConstantU32(call->getArgOperand(3));
        llvm::IRBuilder<> at(call);
        if (col < 3 && numbers[col] != ~0u)
          call->setArgOperand(4, at.CreateLoad(f32, slots_out[col]));
        else if (col == 3 && !keep_alpha)
          call->setArgOperand(4, llvm::ConstantFP::get(f32, 1.0));
      }
}

// Moves the DXIL entry's blocks to the end of the function being built, turns its returns into branches to
// `epilogue`, lowers every dx.op call, and returns the block that enters the body. The insertion point is kept (the
// caller branches to the body from where its prologue ended; the prologue may have added blocks, e.g. vertex pulls).
llvm::Expected<llvm::BasicBlock *>
LowerBody(const DXILShader &dxil, struct context &ctx, llvm::Function *dxil_main, llvm::BasicBlock *epilogue,
          const EntryInfo *vertex_outputs = nullptr) {
  ApplyProbe(dxil, *dxil_main);
  auto ip = ctx.builder.saveIP();
  auto body = &dxil_main->getEntryBlock();
  std::vector<llvm::BasicBlock *> blocks;
  for (auto &bb : *dxil_main)
    blocks.push_back(&bb);
  ctx.function->getBasicBlockList().splice(ctx.function->end(), dxil_main->getBasicBlockList());
  std::vector<llvm::CallInst *> calls; // collected first: lowering erases them
  // DXC marks most float math `fast`. D3D keeps NaN and infinity, so no stage assumes them away, and compares keep no
  // fast flag at all (Metal's compiler treats any as leave to ignore NaN: (x < y) || (x >= y) came out true for a NaN).
  // As airconv's DXBC path (Converter::UseFastMath), pre-raster stages neither reassociate, fuse nor use reciprocals,
  // so a depth prepass and a base pass compute the same positions (their depth EQUAL test holds). (MacNeutron)
  bool pre_raster = dxil.entry.kind == ShaderKind::Vertex || dxil.entry.kind == ShaderKind::Geometry ||
                    dxil.entry.kind == ShaderKind::Hull || dxil.entry.kind == ShaderKind::Domain;
  for (auto *bb : blocks) {
    if (auto ret = llvm::dyn_cast_or_null<llvm::ReturnInst>(bb->getTerminator())) {
      llvm::BranchInst::Create(epilogue, ret);
      ret->eraseFromParent();
    }
    for (auto &inst : *bb) {
      if (llvm::isa<llvm::FCmpInst>(&inst)) {
        inst.copyFastMathFlags(llvm::FastMathFlags());
      } else if (llvm::isa<llvm::FPMathOperator>(&inst)) {
        inst.setHasNoNaNs(false);
        inst.setHasNoInfs(false);
        if (pre_raster && DxilVsFast() != 1) {
          inst.setHasAllowReassoc(false);
          inst.setHasAllowContract(false);
          inst.setHasAllowReciprocal(false);
        }
      }
      if (auto call = llvm::dyn_cast<llvm::CallInst>(&inst); call && OpCode(*call) != ~0u)
        calls.push_back(call);
    }
  }
  Lowering lowering(dxil, ctx, vertex_outputs);
  for (auto *call : calls)
    if (auto err = lowering.Lower(call))
      return std::move(err);
  lowering.Cleanup(*ctx.function);
  ctx.builder.restoreIP(ip);
  return body;
}

// Removes what's left of DXIL once the AIR function is built, and checks the module.
llvm::Error Finish(llvm::Module &module, llvm::Function *dxil_main) {
  dxil_main->eraseFromParent();
  for (auto it = module.begin(); it != module.end();) {
    auto &fn = *it++;
    if (fn.getName().startswith("dx.op.") && fn.use_empty())
      fn.eraseFromParent();
  }
  if (auto err = MoveConstantsToConstantSpace(module))
    return err;
  StripDXIL(module);
  std::string problems;
  llvm::raw_string_ostream os(problems);
  if (llvm::verifyModule(module, &os))
    return llvm::make_error<UnsupportedFeature>("DXIL: invalid module after lowering: " + os.str());
  return llvm::Error::success();
}

} // namespace

llvm::Expected<std::unique_ptr<llvm::Module>>
ConvertDXIL(SM50ShaderInternal *shader, const char *name, llvm::LLVMContext &context, SM50_SHADER_COMPILATION_ARGUMENT_DATA *pArgs) {
  auto &dxil = *shader->dxil;
  auto &entry = dxil.entry;
  if (entry.kind == ShaderKind::Geometry)
    return llvm::make_error<UnsupportedFeature>("DXIL: a geometry shader compiles only with its vertex shader");
  auto loaded = Load(dxil, context);
  if (!loaded)
    return loaded.takeError();
  auto [module, dxil_main] = std::move(*loaded);

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
  SM50_SHADER_IA_INPUT_LAYOUT_DATA *ia_layout = nullptr;
  args_get_data<SM50_SHADER_IA_INPUT_LAYOUT, SM50_SHADER_IA_INPUT_LAYOUT_DATA>(pArgs, &ia_layout);
  SM50_SHADER_PSO_PIXEL_SHADER_DATA *pso = nullptr;
  args_get_data<SM50_SHADER_PSO_PIXEL_SHADER, SM50_SHADER_PSO_PIXEL_SHADER_DATA>(pArgs, &pso);
  uint32_t pso_sample_mask = pso ? pso->sample_mask : 0xffffffff;
  bool vertex = entry.kind == ShaderKind::Vertex, pixel = entry.kind == ShaderKind::Pixel;

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
    sig_ctx.ia_layout = ia_layout;
    if (pso) { // as convert_dxbc_pixel_shader
      sig_ctx.dual_source_blending = pso->dual_source_blending;
      sig_ctx.disable_depth_output = pso->disable_depth_output;
      sig_ctx.unorm_output_reg_mask = pso->unorm_output_reg_mask;
      if (pso->pixel_formats)
        memcpy(sig_ctx.pixel_formats, pso->pixel_formats, sizeof(sig_ctx.pixel_formats));
    }
    for (auto &p : shader->signature_handlers)
      p(sig_ctx);
  }
  if (pixel && pso_sample_mask != 0xffffffff) { // as convert_dxbc_pixel_shader
    auto index = func_signature.DefineOutput(air::OutputCoverageMask{});
    epilogue >> [=](pvalue value) -> IRValue {
      return make_irvalue([=](struct context ctx) {
        if (ctx.resource.coverage_mask_reg)
          return value;
        return ctx.builder.CreateInsertValue(value, ctx.builder.getInt32(ctx.pso_sample_mask), {index});
      });
    };
  }
  // Vertex shaders: IDs as convert_dxbc_vertex_shader defines them.
  uint32_t vertex_idx = 0, base_vertex_idx = 0, instance_idx = 0, base_instance_idx = 0;
  if (vertex) {
    vertex_idx = func_signature.DefineInput(air::InputVertexID{});
    base_vertex_idx = func_signature.DefineInput(air::InputBaseVertex{});
    instance_idx = func_signature.DefineInput(air::InputInstanceID{});
    base_instance_idx = func_signature.DefineInput(air::InputBaseInstance{});
  }
  auto binding_map = rootsig ? setup_binding_rootsig(shader_info, func_signature, *module, shader->shader_type,
                                                     rootsig->bytecode, rootsig->bytecode_length)
                             : setup_binding_table2(shader_info, func_signature, *module);

  auto [function, function_metadata] = func_signature.CreateFunction(name, context, *module, 0, false);
  if (DxilVsFast() == 2 && (dxil.entry.kind == ShaderKind::Vertex || dxil.entry.kind == ShaderKind::Geometry ||
                            dxil.entry.kind == ShaderKind::Hull || dxil.entry.kind == ShaderKind::Domain))
    for (auto attr : {"invariance-late-contract", "invariance-late-reassoc", "invariance-late-unsafe-fp-math"})
      function->addFnAttr(attr);
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
  if (vertex) {
    resource_map.vertex_id_with_base = function->getArg(vertex_idx);
    resource_map.base_vertex_id = function->getArg(base_vertex_idx);
    resource_map.instance_id_with_base = function->getArg(instance_idx);
    resource_map.base_instance_id = function->getArg(base_instance_idx);
    resource_map.vertex_id = builder.CreateSub(resource_map.vertex_id_with_base, resource_map.base_vertex_id);
    resource_map.instance_id = builder.CreateSub(resource_map.instance_id_with_base, resource_map.base_instance_id);
  }
  if (vertex || pixel) { // the register files loadInput/storeOutput address (element row, column)
    auto files = [&](register_file &file, uint32_t count) {
      file.ptr_int4 = builder.CreateAlloca(llvm::ArrayType::get(types._int4, std::max(count, 1u)));
      file.ptr_float4 = builder.CreateBitCast(file.ptr_int4, llvm::ArrayType::get(types._float4, std::max(count, 1u))->getPointerTo());
    };
    files(resource_map.input, shader->max_input_register);
    files(resource_map.output, shader->max_output_register);
    resource_map.input_element_count = shader->max_input_register;
    resource_map.output_element_count = shader->max_output_register;
  }

  struct context ctx{
      .builder = builder,
      .air = air,
      .binding = *binding_map,
      .llvm = context,
      .module = *module,
      .function = function,
      .resource = resource_map,
      .types = types,
      .pso_sample_mask = pso_sample_mask,
      .shader_type = shader->shader_type,
      .metal_version = metal_version,
  };
  if (auto err = prologue.build(ctx).takeError())
    return std::move(err);

  auto body = LowerBody(dxil, ctx, dxil_main, epilogue_bb);
  if (!body)
    return body.takeError();
  builder.CreateBr(*body);

  builder.SetInsertPoint(epilogue_bb);
  auto result = epilogue.build(ctx);
  if (auto err = result.takeError())
    return std::move(err);
  if (*result)
    builder.CreateRet(*result);
  else
    builder.CreateRetVoid();

  module->getOrInsertNamedMetadata(vertex ? "air.vertex" : pixel ? "air.fragment" : "air.kernel")->addOperand(function_metadata);
  if (auto err = Finish(*module, dxil_main))
    return std::move(err);
  return std::move(module);
}

llvm::Expected<std::unique_ptr<llvm::Module>>
ConvertDXILGeometryPipeline(bool object, SM50ShaderInternal *vs, SM50ShaderInternal *gs, const char *name,
                            llvm::LLVMContext &context, SM50_SHADER_COMPILATION_ARGUMENT_DATA *pArgs) {
  if (!vs->dxil || !gs->dxil || vs->dxil->entry.kind != ShaderKind::Vertex || gs->dxil->entry.kind != ShaderKind::Geometry)
    return llvm::make_error<UnsupportedFeature>("DXIL: a geometry pipeline needs a DXIL vertex and geometry shader");
  auto stage = object ? vs : gs;
  auto loaded = Load(*stage->dxil, context);
  if (!loaded)
    return loaded.takeError();
  auto module = std::move(loaded->first);
  auto dxil_main = loaded->second;
  // The object function runs the vertex shader into the payload; the mesh function runs the geometry shader on it,
  // whose inputs are relinked to the vertex shader's outputs by semantic (as the varyings between stages).
  ShaderBody body = [&](struct context &ctx, llvm::BasicBlock *epilogue) {
    return LowerBody(*stage->dxil, ctx, dxil_main, epilogue, object ? nullptr : &vs->dxil->entry);
  };
  if (auto err = object ? convert_dxbc_vertex_for_geometry_shader(vs, name, gs, context, *module, pArgs, body)
                        : convert_dxbc_geometry_shader(gs, name, vs, context, *module, pArgs, body))
    return std::move(err);
  if (auto err = Finish(*module, dxil_main))
    return std::move(err);
  return std::move(module);
}

} // namespace dxmt::dxil
