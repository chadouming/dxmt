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

// Entry-point inputs and outputs for DXIL shaders, built from the entry's signature metadata with airconv's IO helpers
// (as dxbc_signature.cpp builds them from DXBC dcl_* instructions). Elements live in airconv's register file at
// (start_row + row, start_col + col); varyings between stages are named by semantic and carry their components from x.
namespace dxmt::dxil {

using namespace dxmt::dxbc;

namespace {

llvm::Error Unsupported(const std::string &what) {
  return llvm::make_error<UnsupportedFeature>("DXIL: " + what + " not supported");
}

RegisterComponentType RegType(ComponentType t) {
  switch (t) {
  case ComponentType::I16:
  case ComponentType::I32:
  case ComponentType::I64: return RegisterComponentType::Int;
  case ComponentType::I1:
  case ComponentType::U16:
  case ComponentType::U32:
  case ComponentType::U64: return RegisterComponentType::Uint;
  default: return RegisterComponentType::Float;
  }
}

air::Interpolation ToInterpolation(Interpolation i) {
  switch (i) {
  case Interpolation::Constant: return air::Interpolation::flat;
  case Interpolation::LinearCentroid: return air::Interpolation::centroid_perspective;
  case Interpolation::LinearNoperspective: return air::Interpolation::center_no_perspective;
  case Interpolation::LinearNoperspectiveCentroid: return air::Interpolation::centroid_no_perspective;
  case Interpolation::LinearSample: return air::Interpolation::sample_perspective;
  case Interpolation::LinearNoperspectiveSample: return air::Interpolation::sample_no_perspective;
  default: return air::Interpolation::center_perspective;
  }
}

uint32_t Mask(const SignatureElement &e) {
  return ((1u << e.cols) - 1) << e.start_col;
}

// Vertex output: register `reg`'s components [col, col + cols) become the varying's x, y, ...
std::function<IRValue(pvalue)> PopShifted(uint32_t reg, uint32_t col, uint32_t cols, uint32_t to_element) {
  return [=](pvalue ret) {
    return make_irvalue([=](struct context ctx) -> pvalue {
      auto &b = ctx.builder;
      auto array = ctx.resource.output.ptr_int4;
      auto array_ty = llvm::cast<llvm::PointerType>(array->getType())->getNonOpaquePointerElementType();
      auto row = b.CreateLoad(ctx.types._int4, b.CreateGEP(array_ty, array, {b.getInt32(0), b.getInt32(reg)}));
      int idx[4];
      for (uint32_t i = 0; i < 4; i++)
        idx[i] = (int)(i < cols ? col + i : col);
      auto shifted = b.CreateShuffleVector(row, idx);
      auto field_ty = ctx.function->getReturnType()->getStructElementType(to_element);
      return b.CreateInsertValue(ret, b.CreateBitCast(shifted, field_ty), {to_element});
    });
  };
}

// Pixel input: the varying's x, y, ... land at register `reg`'s components [col, col + cols).
IREffect InitShifted(uint32_t arg_index, uint32_t reg, uint32_t col, uint32_t cols) {
  return make_effect([=](struct context ctx) {
    auto &b = ctx.builder;
    auto array = ctx.resource.input.ptr_int4;
    auto array_ty = llvm::cast<llvm::PointerType>(array->getType())->getNonOpaquePointerElementType();
    auto arg = b.CreateBitCast(ctx.function->getArg(arg_index), ctx.types._int4);
    for (uint32_t i = 0; i < cols; i++)
      b.CreateStore(b.CreateExtractElement(arg, i),
                    b.CreateGEP(array_ty, array, {b.getInt32(0), b.getInt32(reg), b.getInt32(col + i)}));
    return std::monostate{};
  });
}

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

void Grow(uint32_t &max_register, const SignatureElement &e) {
  if (e.start_row >= 0)
    max_register = std::max(max_register, (uint32_t)e.start_row + e.rows);
}

llvm::Error
AddComputeHandlers(const llvm::Module &module, SM50ShaderInternal *shader) {
  // Compute IDs aren't signature elements in DXIL: define the inputs the module's dx.op calls read.
  if (ModuleCallsOp(module, op::ThreadId))
    DefineSpecial<air::InputThreadPositionInGrid>(shader, &io_binding_map::thread_id_arg);
  if (ModuleCallsOp(module, op::GroupId))
    DefineSpecial<air::InputThreadgroupPositionInGrid>(shader, &io_binding_map::thread_group_id_arg);
  if (ModuleCallsOp(module, op::ThreadIdInGroup))
    DefineSpecial<air::InputThreadPositionInThreadgroup>(shader, &io_binding_map::thread_id_in_group_arg);
  if (ModuleCallsOp(module, op::FlattenedThreadIdInGroup))
    DefineSpecial<air::InputThreadIndexInThreadgroup>(shader, &io_binding_map::thread_id_in_group_flat_arg);
  return llvm::Error::success();
}

llvm::Error
AddVertexHandlers(const EntryInfo &entry, SM50ShaderInternal *shader, std::map<uint32_t, uint32_t> &unpacked) {
  auto &handlers = shader->signature_handlers;
  auto &fs = shader->func_signature;
  for (auto &e : entry.inputs) {
    if (e.kind == SemanticKind::VertexID || e.kind == SemanticKind::InstanceID) {
      bool vertex = e.kind == SemanticKind::VertexID;
      if (e.start_row < 0) { // read from io_binding_map's vertex_id/instance_id (the converter computes them)
        unpacked[e.id] = vertex ? kUnpackedVertexID : kUnpackedInstanceID;
        continue;
      }
      uint32_t reg = e.start_row, mask = Mask(e);
      handlers.push_back([=](SignatureContext &sig) {
        sig.prologue << make_effect_bind([=](struct context ctx) {
          return store_at_vec4_array_masked(ctx.resource.input.ptr_int4, ctx.builder.getInt32(reg),
                                            vertex ? ctx.resource.vertex_id : ctx.resource.instance_id, mask);
        });
      });
      Grow(shader->max_input_register, e);
      continue;
    }
    if (e.kind != SemanticKind::Arbitrary)
      return Unsupported("vertex shader input " + e.name);
    for (uint32_t r = 0; r < e.rows; r++) {
      uint32_t reg = e.start_row + r, mask = Mask(e);
      auto type = (air::InputAttributeComponentType)RegType(e.type);
      auto name = UserName(e, r);
      handlers.push_back([=](SignatureContext &sig) { // as DCL_INPUT in handle_signature_vs
        if (sig.ia_layout) {
          for (unsigned i = 0; i < sig.ia_layout->num_elements; i++)
            if (sig.ia_layout->elements[i].reg == reg) {
              sig.prologue << pull_vertex_input(sig.func_signature, reg, mask, sig.ia_layout->elements[i],
                                                sig.ia_layout->slot_mask);
              break;
            }
        } else {
          auto index = sig.func_signature.DefineInput(air::InputVertexStageIn{.attribute = reg, .type = type, .name = name});
          sig.prologue << init_input_reg(index, reg, mask);
        }
      });
    }
    Grow(shader->max_input_register, e);
  }
  for (auto &e : entry.outputs) {
    uint32_t reg = e.start_row, mask = Mask(e);
    switch (e.kind) {
    case SemanticKind::Position: {
      auto index = fs.DefineOutput(air::OutputPosition{.type = air::msl_float4});
      handlers.push_back([=](SignatureContext &sig) {
        if (!sig.skip_vertex_output)
          sig.epilogue >> pop_output_reg_sanitize_pos(reg, mask, index);
      });
      break;
    }
    case SemanticKind::RenderTargetArrayIndex:
    case SemanticKind::ViewPortArrayIndex: {
      auto index = e.kind == SemanticKind::RenderTargetArrayIndex ? fs.DefineOutput(air::OutputRenderTargetArrayIndex{})
                                                                  : fs.DefineOutput(air::OutputViewportArrayIndex{});
      handlers.push_back([=](SignatureContext &sig) {
        if (!sig.skip_vertex_output)
          sig.epilogue >> pop_output_reg(reg, mask, index);
      });
      break;
    }
    case SemanticKind::Arbitrary:
      for (uint32_t r = 0; r < e.rows; r++) {
        auto index = fs.DefineOutput(Varying(e, r));
        uint32_t col = e.start_col, cols = e.cols;
        handlers.push_back([=](SignatureContext &sig) {
          if (!sig.skip_vertex_output)
            sig.epilogue >> PopShifted(reg + r, col, cols, index);
        });
      }
      break;
    default:
      return Unsupported("vertex shader output " + e.name);
    }
    Grow(shader->max_output_register, e);
  }
  return llvm::Error::success();
}

llvm::Error
AddPixelHandlers(const EntryInfo &entry, SM50ShaderInternal *shader, std::map<uint32_t, uint32_t> &unpacked) {
  auto &handlers = shader->signature_handlers;
  auto &fs = shader->func_signature;
  for (auto &e : entry.inputs) {
    uint32_t reg = e.start_row, mask = Mask(e);
    switch (e.kind) {
    case SemanticKind::Position: {
      auto index = fs.DefineInput(air::InputPosition{.interpolation = ToInterpolation(e.interpolation)});
      handlers.push_back([=](SignatureContext &sig) { sig.prologue << init_input_reg(index, reg, mask, true); });
      break;
    }
    case SemanticKind::IsFrontFace:
    case SemanticKind::SampleIndex:
    case SemanticKind::PrimitiveID:
    case SemanticKind::Coverage:
    case SemanticKind::RenderTargetArrayIndex:
    case SemanticKind::ViewPortArrayIndex: {
      uint32_t index = e.kind == SemanticKind::IsFrontFace              ? fs.DefineInput(air::InputFrontFacing{})
                       : e.kind == SemanticKind::SampleIndex            ? fs.DefineInput(air::InputSampleIndex{})
                       : e.kind == SemanticKind::PrimitiveID            ? fs.DefineInput(air::InputPrimitiveID{})
                       : e.kind == SemanticKind::RenderTargetArrayIndex ? fs.DefineInput(air::InputRenderTargetArrayIndex{})
                       : e.kind == SemanticKind::ViewPortArrayIndex     ? fs.DefineInput(air::InputViewportArrayIndex{})
                                                                        : fs.DefineInput(air::InputInputCoverage{});
      if (e.start_row < 0)
        unpacked[e.id] = index;
      else
        handlers.push_back([=](SignatureContext &sig) { sig.prologue << init_input_reg(index, reg, mask); });
      break;
    }
    case SemanticKind::Arbitrary:
      for (uint32_t r = 0; r < e.rows; r++) {
        auto index = fs.DefineInput(air::InputFragmentStageIn{
            .user = Varying(e, r).user,
            .type = Varying(e, r).type,
            .interpolation = ToInterpolation(e.interpolation),
            .pull_mode = false
        });
        uint32_t col = e.start_col, cols = e.cols;
        handlers.push_back([=](SignatureContext &sig) { sig.prologue << InitShifted(index, reg + r, col, cols); });
      }
      break;
    default:
      return Unsupported("pixel shader input " + e.name);
    }
    Grow(shader->max_input_register, e);
  }
  for (auto &e : entry.outputs) {
    uint32_t reg = e.start_row, mask = Mask(e);
    switch (e.kind) {
    case SemanticKind::Target: { // as DCL_OUTPUT in handle_signature_ps, with the semantic index as the target
      uint32_t n = e.semantic_indices.empty() ? 0 : e.semantic_indices[0];
      shader->pso_valid_output_reg_mask |= 1u << n;
      auto sig_type = RegType(e.type);
      handlers.push_back([=](SignatureContext &sig) {
        auto type = component_type_from_pixel_format(sig.pixel_formats[n]);
        if (sig.dual_source_blending || type == RegisterComponentType::Unknown)
          type = sig_type;
        if (sig.dual_source_blending && n > 1)
          return;
        auto index = sig.func_signature.DefineOutput(air::OutputRenderTarget{
            .dual_source_blending = sig.dual_source_blending, .index = n, .type = to_msl_type(type)});
        if (type == RegisterComponentType::Float && (sig.unorm_output_reg_mask & (1u << n)))
          sig.epilogue >> pop_output_reg_fix_unorm(reg, mask, index);
        else
          sig.epilogue >> pop_output_reg(reg, mask, index);
      });
      break;
    }
    case SemanticKind::Depth:
    case SemanticKind::DepthLessEqual:
    case SemanticKind::DepthGreaterEqual: { // as the OUTPUT_DEPTH case in handle_signature_ps
      auto argument = e.kind == SemanticKind::DepthGreaterEqual ? air::DepthArgument::greater
                      : e.kind == SemanticKind::DepthLessEqual  ? air::DepthArgument::less
                                                                : air::DepthArgument::any;
      handlers.push_back([=](SignatureContext &sig) {
        sig.prologue << make_effect([](struct context ctx) -> std::monostate {
          ctx.resource.depth_output_reg = ctx.builder.CreateAlloca(ctx.types._float);
          return {};
        });
        if (sig.disable_depth_output)
          return;
        auto index = sig.func_signature.DefineOutput(air::OutputDepth{.depth_argument = argument});
        sig.epilogue >> [=](pvalue v) {
          return make_irvalue([=](struct context ctx) {
            return ctx.builder.CreateInsertValue(v, ctx.builder.CreateLoad(ctx.types._float, ctx.resource.depth_output_reg), {index});
          });
        };
      });
      break;
    }
    case SemanticKind::Coverage: {
      auto index = fs.DefineOutput(air::OutputCoverageMask{});
      handlers.push_back([=](SignatureContext &sig) {
        sig.prologue << make_effect([](struct context ctx) -> std::monostate {
          ctx.resource.coverage_mask_reg = ctx.builder.CreateAlloca(ctx.types._int);
          return {};
        });
        sig.epilogue >> [=](pvalue v) {
          return make_irvalue([=](struct context ctx) {
            auto mask = ctx.builder.CreateLoad(ctx.types._int, ctx.resource.coverage_mask_reg);
            return ctx.builder.CreateInsertValue(
                v, ctx.pso_sample_mask != 0xffffffff ? ctx.builder.CreateAnd(mask, ctx.pso_sample_mask) : mask, {index});
          });
        };
      });
      shader->ps_has_coverage_output = 1;
      break;
    }
    case SemanticKind::StencilRef: {
      auto index = fs.DefineOutput(air::OutputStencilRef{});
      handlers.push_back([=](SignatureContext &sig) {
        sig.prologue << make_effect([](struct context ctx) -> std::monostate {
          ctx.resource.stencil_ref_reg = ctx.builder.CreateAlloca(ctx.types._int);
          return {};
        });
        sig.epilogue >> [=](pvalue v) {
          return make_irvalue([=](struct context ctx) {
            return ctx.builder.CreateInsertValue(v, ctx.builder.CreateLoad(ctx.types._int, ctx.resource.stencil_ref_reg), {index});
          });
        };
      });
      break;
    }
    default:
      return Unsupported("pixel shader output " + e.name);
    }
    Grow(shader->max_output_register, e);
  }
  return llvm::Error::success();
}

} // namespace

air::OutputVertex Varying(const SignatureElement &e, uint32_t row) {
  return air::OutputVertex{.user = UserName(e, row), .type = to_msl_type(RegType(e.type))};
}

void
AddWaveInputs(const llvm::Module &module, SM50ShaderInternal *shader, uint32_t &lane_index_arg, uint32_t &lane_count_arg) {
  for (uint32_t opcode : {op::WaveIsFirstLane, op::WaveGetLaneIndex, op::WaveGetLaneCount, op::WaveAnyTrue, op::WaveAllTrue,
                          op::WaveActiveAllEqual, op::WaveActiveBallot, op::WaveReadLaneAt, op::WaveReadLaneFirst,
                          op::WaveActiveOp, op::WaveActiveBit, op::WavePrefixOp, op::WaveAllBitCount, op::WavePrefixBitCount})
    if (ModuleCallsOp(module, opcode)) {
      lane_index_arg = shader->func_signature.DefineInput(air::InputThreadIndexInSimdgroup{});
      lane_count_arg = shader->func_signature.DefineInput(air::InputThreadsPerSimdgroup{});
      return;
    }
}

llvm::Error
AddSignatureHandlers(const EntryInfo &entry, const llvm::Module &module, SM50ShaderInternal *shader,
                     std::map<uint32_t, uint32_t> &unpacked_inputs) {
  switch (entry.kind) {
  case ShaderKind::Compute: return AddComputeHandlers(module, shader);
  case ShaderKind::Vertex: return AddVertexHandlers(entry, shader, unpacked_inputs);
  case ShaderKind::Pixel: return AddPixelHandlers(entry, shader, unpacked_inputs);
  default: return Unsupported("this shader stage");
  }
}

} // namespace dxmt::dxil
