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
#include "llvm/IR/Instructions.h"

namespace dxmt::dxil {

using namespace dxmt::dxbc;
using shader::common::ResourceType;
using shader::common::ScalerDataType;

namespace {

ResourceType ToResourceType(ResourceKind kind) {
  switch (kind) {
  case ResourceKind::Texture1D: return ResourceType::Texture1D;
  case ResourceKind::Texture1DArray: return ResourceType::Texture1DArray;
  case ResourceKind::Texture2D: return ResourceType::Texture2D;
  case ResourceKind::Texture2DArray: return ResourceType::Texture2DArray;
  case ResourceKind::Texture2DMS: return ResourceType::Texture2DMultisampled;
  case ResourceKind::Texture2DMSArray: return ResourceType::Texture2DMultisampledArray;
  case ResourceKind::Texture3D: return ResourceType::Texture3D;
  case ResourceKind::TextureCube: return ResourceType::TextureCube;
  case ResourceKind::TextureCubeArray: return ResourceType::TextureCubeArray;
  case ResourceKind::TypedBuffer: return ResourceType::TextureBuffer;
  default: return ResourceType::NonApplicable; // raw and structured buffers
  }
}

ScalerDataType ToScalerType(ComponentType type) {
  switch (type) {
  case ComponentType::I16:
  case ComponentType::I32:
  case ComponentType::I64: return ScalerDataType::Int;
  case ComponentType::U16:
  case ComponentType::U32:
  case ComponentType::U64: return ScalerDataType::Uint;
  default: return ScalerDataType::Float;
  }
}

struct Usage {
  bool read = false, written = false, sampled = false, compared = false;
};

// What each resource is used for: airconv's binding maps pick Metal access modes and depth textures from it.
std::map<std::pair<ResourceClass, uint32_t>, Usage>
ScanUsage(const EntryInfo &entry, const llvm::Module &module) {
  std::map<std::pair<ResourceClass, uint32_t>, Usage> usage;
  for (auto &fn : module)
    if (fn.getName().startswith("dx.op."))
      for (auto *user : fn.users()) {
        auto call = llvm::dyn_cast<llvm::CallInst>(user);
        if (!call || call->arg_size() < 2)
          continue;
        uint32_t opcode = OpCode(*call);
        auto handle = ResolveHandle(entry, call->getArgOperand(1));
        if (!handle) {
          llvm::consumeError(handle.takeError());
          continue;
        }
        auto &u = usage[{handle->cls, handle->range}];
        switch (opcode) {
        case op::Sample: case op::SampleBias: case op::SampleLevel: case op::SampleGrad: case op::TextureGather:
        case op::CalculateLOD:
          u.sampled = true;
          break;
        case op::SampleCmp: case op::SampleCmpLevelZero: case op::TextureGatherCmp:
          u.sampled = u.compared = true;
          break;
        case op::TextureLoad: case op::BufferLoad: case op::RawBufferLoad: case op::CBufferLoad: case op::CBufferLoadLegacy:
          u.read = true;
          break;
        case op::TextureStore: case op::BufferStore: case op::RawBufferStore:
          u.written = true;
          break;
        case op::AtomicBinOp: case op::AtomicCompareExchange:
          u.read = u.written = true;
          break;
        }
      }
  return usage;
}

// airconv's resource maps (ShaderInfo), keyed by range ID, as the DXBC dcl_* declarations fill them.
void FillResourceMaps(const EntryInfo &entry, const llvm::Module &module, ShaderInfo &info) {
  auto usage = ScanUsage(entry, module);
  for (auto &r : entry.resources) {
    ResourceRange range{r.id, r.lower_bound, r.size, r.space};
    auto u = usage[{r.cls, r.id}];
    switch (r.cls) {
    case ResourceClass::SRV: {
      ShaderResourceViewInfo srv{range, ToScalerType(r.element_type), ToResourceType(r.kind)};
      srv.read = u.read;
      srv.sampled = u.sampled;
      srv.compared = u.compared;
      srv.structure_stride = r.stride;
      srv.arg_index = srv.arg_metadata_index = ~0u;
      info.srvMap[r.id] = srv;
      break;
    }
    case ResourceClass::UAV: {
      UnorderedAccessViewInfo uav{range, ToScalerType(r.element_type), ToResourceType(r.kind)};
      uav.read = u.read;
      uav.written = u.written;
      uav.global_coherent = r.globally_coherent;
      uav.rasterizer_order = r.rasterizer_ordered;
      uav.with_counter = r.has_counter;
      uav.structure_stride = r.stride;
      uav.arg_index = uav.arg_metadata_index = uav.arg_counter_index = ~0u;
      info.uavMap[r.id] = uav;
      break;
    }
    case ResourceClass::CBuffer:
      info.cbufferMap[r.id] = ConstantBufferInfo{range, (r.cbuffer_size + 15) / 16, ~0u};
      break;
    case ResourceClass::Sampler:
      info.samplerMap[r.id] = SamplerInfo{range, ~0u, ~0u, ~0u};
      break;
    }
  }
}

} // namespace

llvm::Error
InitializeDXIL(const Container &container, SM50ShaderInternal *shader, MTL_SHADER_REFLECTION *refl) {
  llvm::LLVMContext context;
  context.setOpaquePointers(false);
  auto module = LoadModule(context, container.bitcode, container.bitcode_size);
  if (!module)
    return module.takeError();
  auto entry = ReadEntry(**module);
  if (!entry)
    return entry.takeError();
  switch (entry->kind) {
  case ShaderKind::Vertex: shader->shader_type = microsoft::D3D10_SB_VERTEX_SHADER; break;
  case ShaderKind::Pixel: shader->shader_type = microsoft::D3D10_SB_PIXEL_SHADER; break;
  case ShaderKind::Compute: shader->shader_type = microsoft::D3D11_SB_COMPUTE_SHADER; break;
  default: return llvm::make_error<UnsupportedFeature>("DXIL: only vertex, pixel and compute shaders are supported");
  }
  FillResourceMaps(*entry, **module, shader->shader_info);
  std::map<uint32_t, uint32_t> unpacked_inputs;
  if (auto err = AddSignatureHandlers(*entry, **module, shader, unpacked_inputs))
    return err;
  uint32_t lane_index_arg = ~0u, lane_count_arg = ~0u;
  AddWaveInputs(**module, shader, lane_index_arg, lane_count_arg);
  if (entry->kind == ShaderKind::Compute) {
    std::copy(entry->numthreads, entry->numthreads + 3, shader->threadgroup_size);
    shader->func_signature.UseMaxWorkgroupSize(entry->numthreads[0] * entry->numthreads[1] * entry->numthreads[2]);
  }
  auto dxil = std::make_shared<DXILShader>();
  dxil->bitcode.assign(container.bitcode, container.bitcode + container.bitcode_size);
  dxil->entry = std::move(*entry);
  dxil->unpacked_inputs = std::move(unpacked_inputs);
  dxil->lane_index_arg = lane_index_arg;
  dxil->lane_count_arg = lane_count_arg;
  shader->dxil = dxil;
  if (refl) {
    *refl = {};
    refl->ConstanttBufferTableBindIndex = ~0u;
    refl->ArgumentBufferBindIndex = ~0u;
    if (shader->shader_type == microsoft::D3D11_SB_COMPUTE_SHADER)
      std::copy(shader->threadgroup_size, shader->threadgroup_size + 3, refl->ThreadgroupSize);
    if (shader->shader_type == microsoft::D3D10_SB_PIXEL_SHADER) {
      refl->PixelShader.ValidRenderTargets = shader->pso_valid_output_reg_mask;
      refl->PixelShader.HasCoverageOutput = shader->ps_has_coverage_output;
    }
    refl->NumOutputElement = shader->max_output_register;
  }
  return llvm::Error::success();
}

} // namespace dxmt::dxil
