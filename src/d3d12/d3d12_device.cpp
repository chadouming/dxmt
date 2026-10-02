/*
 * Copyright 2026 Feifan He for CodeWeavers
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

#include "d3d12_device.hpp"
#include "d3d12_stats.hpp"
#include "d3d12_feature_data.hpp"
#include "d3d12_dxil_dump.hpp"
#include "d3d12_device_child.hpp"
#include "Metal.hpp"
#include "com/com_pointer.hpp"
#include "com/com_object.hpp"
#include "dxgi_interfaces.h"
#include "dxmt_format.hpp"
#include "log/log.hpp"
#include <map>
#include <mutex>
#include "d3d10_1.h"
#include "d3d11_4.h"

namespace dxmt {

const GUID kD3D12DeviceDownlevelUUID = {0x74eaee3f, 0x2f4b, 0x476d, {0x82, 0xba, 0x2b, 0x85, 0xcb, 0x49, 0xe3, 0x10}};

HRESULT PopulateWMTTextureInfo(WMT::Device Device, WMTTextureInfo &InfoOut, const D3D12_RESOURCE_DESC &Desc);

// Fills either description from a pipeline stream, with D3D12's defaults for what the stream leaves out (MacNeutron:
// shared by CreatePipelineState and pipeline libraries' LoadPipeline).
HRESULT
ParsePipelineStream(
    const D3D12_PIPELINE_STATE_STREAM_DESC *pDesc, D3D12_GRAPHICS_PIPELINE_STATE_DESC &desc_graphics,
    D3D12_COMPUTE_PIPELINE_STATE_DESC &desc_cs, bool &compute
) {
  desc_graphics = {};
  desc_cs = {};
  const char *stream_start = reinterpret_cast<const char *>(pDesc->pPipelineStateSubobjectStream);
  const char *stream_end = stream_start + pDesc->SizeInBytes;

  {
    desc_graphics.DepthStencilState.DepthEnable = TRUE;
    desc_graphics.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    desc_graphics.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    desc_graphics.DepthStencilState.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    desc_graphics.DepthStencilState.FrontFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
    desc_graphics.DepthStencilState.FrontFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
    desc_graphics.DepthStencilState.FrontFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
    desc_graphics.DepthStencilState.BackFace = desc_graphics.DepthStencilState.FrontFace;
    desc_graphics.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc_graphics.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
    desc_graphics.RasterizerState.DepthClipEnable = TRUE;
    desc_graphics.RasterizerState.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
    desc_graphics.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    desc_graphics.SampleDesc.Count = 1;
    desc_graphics.SampleDesc.Quality = 0;
    desc_graphics.SampleMask = D3D12_DEFAULT_SAMPLE_MASK;
  }

  uint32_t defined_type = 0;

  while (stream_start < stream_end) {
    if (stream_start + sizeof(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE) > stream_end) {
      ERR("CreatePipelineState: invalid stream");
      return E_INVALIDARG;
    }
    auto type = *reinterpret_cast<const D3D12_PIPELINE_STATE_SUBOBJECT_TYPE *>(stream_start);

    if (defined_type & (1 << type)) {
      ERR("CreatePipelineState: duplicated subobejct type ", type);
      return E_INVALIDARG;
    }
    defined_type |= (1 << type);

#define GET_STREAM_DATA(data_type)                                                                                     \
using subobject_t = struct {                                                                                         \
  D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type;                                                                          \
  data_type data;                                                                                                    \
};                                                                                                                   \
auto subobject = reinterpret_cast<subobject_t const *>(stream_start);                                                \
if (stream_start + sizeof(*subobject) > stream_end) {                                                                \
  ERR("CreatePipelineState: invalid stream");                                                                        \
  return E_INVALIDARG;                                                                                               \
}                                                                                                                    \
stream_start += align(sizeof(*subobject), sizeof(void *));

    switch (type) {
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE: {
      GET_STREAM_DATA(ID3D12RootSignature *);
      desc_cs.pRootSignature = subobject->data;
      desc_graphics.pRootSignature = subobject->data;
      break;
    }
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS: {
      GET_STREAM_DATA(D3D12_SHADER_BYTECODE);
      desc_graphics.VS = subobject->data;
      break;
    }
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS: {
      GET_STREAM_DATA(D3D12_SHADER_BYTECODE);
      desc_graphics.PS = subobject->data;
      break;
    }
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DS: {
      GET_STREAM_DATA(D3D12_SHADER_BYTECODE);
      desc_graphics.DS = subobject->data;
      break;
    }
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_HS: {
      GET_STREAM_DATA(D3D12_SHADER_BYTECODE);
      desc_graphics.HS = subobject->data;
      break;
    }
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS: {
      GET_STREAM_DATA(D3D12_SHADER_BYTECODE);
      desc_graphics.GS = subobject->data;
      break;
    }
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS: {
      GET_STREAM_DATA(D3D12_SHADER_BYTECODE);
      desc_cs.CS = subobject->data;
      break;
    }
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT: {
      GET_STREAM_DATA(D3D12_STREAM_OUTPUT_DESC);
      desc_graphics.StreamOutput = subobject->data;
      break;
    }
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND: {
      GET_STREAM_DATA(D3D12_BLEND_DESC);
      desc_graphics.BlendState = subobject->data;
      break;
    }
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK: {
      GET_STREAM_DATA(UINT);
      desc_graphics.SampleMask = subobject->data;
      break;
    }
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER: {
      GET_STREAM_DATA(D3D12_RASTERIZER_DESC);
      desc_graphics.RasterizerState = subobject->data;
      break;
    }
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL: {
      GET_STREAM_DATA(D3D12_DEPTH_STENCIL_DESC);
      desc_graphics.DepthStencilState = subobject->data;
      break;
    }
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT: {
      GET_STREAM_DATA(D3D12_INPUT_LAYOUT_DESC);
      desc_graphics.InputLayout = subobject->data;
      break;
    }
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE: {
      GET_STREAM_DATA(D3D12_INDEX_BUFFER_STRIP_CUT_VALUE);
      desc_graphics.IBStripCutValue = subobject->data;
      break;
    }
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY: {
      GET_STREAM_DATA(D3D12_PRIMITIVE_TOPOLOGY_TYPE);
      desc_graphics.PrimitiveTopologyType = subobject->data;
      break;
    }
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS: {
      GET_STREAM_DATA(D3D12_RT_FORMAT_ARRAY);
      memcpy(desc_graphics.RTVFormats, subobject->data.RTFormats, sizeof(desc_graphics.RTVFormats));
      desc_graphics.NumRenderTargets = subobject->data.NumRenderTargets;
      break;
    }
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT: {
      GET_STREAM_DATA(DXGI_FORMAT);
      desc_graphics.DSVFormat = subobject->data;
      break;
    }
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC: {
      GET_STREAM_DATA(DXGI_SAMPLE_DESC);
      desc_graphics.SampleDesc = subobject->data;
      break;
    }
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK: {
      GET_STREAM_DATA(UINT);
      desc_graphics.NodeMask = subobject->data;
      desc_cs.NodeMask = subobject->data;
      break;
    }
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO: {
      GET_STREAM_DATA(D3D12_CACHED_PIPELINE_STATE);
      desc_graphics.CachedPSO = subobject->data;
      desc_cs.CachedPSO = subobject->data;
      break;
    }
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS: {
      GET_STREAM_DATA(D3D12_PIPELINE_STATE_FLAGS);
      desc_graphics.Flags = subobject->data;
      desc_cs.Flags = subobject->data;
      break;
    }
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1: {
      GET_STREAM_DATA(D3D12_DEPTH_STENCIL_DESC1);
      desc_graphics.DepthStencilState.StencilEnable = subobject->data.StencilEnable;
      desc_graphics.DepthStencilState.DepthEnable = subobject->data.DepthEnable;
      desc_graphics.DepthStencilState.DepthFunc = subobject->data.DepthFunc;
      desc_graphics.DepthStencilState.DepthWriteMask = subobject->data.DepthWriteMask;
      desc_graphics.DepthStencilState.StencilWriteMask = subobject->data.StencilWriteMask;
      desc_graphics.DepthStencilState.StencilReadMask = subobject->data.StencilReadMask;
      desc_graphics.DepthStencilState.BackFace = subobject->data.BackFace;
      desc_graphics.DepthStencilState.FrontFace = subobject->data.FrontFace;
      if (subobject->data.DepthBoundsTestEnable) {
        WARN("CreatePipelineState: ignore DepthBoundsTestEnable");
      }
      break;
    }
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING: {
      GET_STREAM_DATA(D3D12_VIEW_INSTANCING_DESC);
      if (subobject->data.Flags) {
        WARN("CreatePipelineState: ignore ViewInstancing");
      }
      break;
    }
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS: {
      GET_STREAM_DATA(D3D12_SHADER_BYTECODE);
      if (subobject->data.pShaderBytecode) {
        ERR("CreatePipelineState: unsupported AS");
        return E_NOTIMPL;
      }
      break;
    }
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS: {
      GET_STREAM_DATA(D3D12_SHADER_BYTECODE);
      if (subobject->data.pShaderBytecode) {
        ERR("CreatePipelineState: unsupported MS");
        return E_NOTIMPL;
      }
      break;
    }
    default:
      ERR("CreatePipelineState: unhandled subobject type ", type);
      return E_INVALIDARG;
    }
  }

  compute = desc_cs.CS.pShaderBytecode != nullptr;
  if (compute && (defined_type & (1 << D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS | 1 << D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS |
                                  1 << D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS))) {
    ERR("CreatePipelineState: invalid compute pipeline state stream");
    return E_INVALIDARG;
  }
  return S_OK;
}

class MTLD3D12DeviceImpl : public MTLD3D12Object<ComObject<MTLD3D12Device>> {

  Com<IMTLDXGIAdapter> adapter_;

  bool advertise_numa_ = false;

  dxmt::mutex residency_lock_;
  std::once_flag timestamp_once_;
  uint64_t timestamp_frequency_ = 1;
  dxmt::mutex null_lock_;
  std::map<WMTTextureType, Rc<Texture>> null_textures_;
  Rc<Buffer> null_buffer_;
  BufferViewKey null_buffer_view_;
  WMT::Reference<WMT::ResidencySet> residency_set_;
  std::map<uint64_t, BufferAllocation *> interval_map_;

  InternalCommandLibrary command_library;
  FormatCapabilityInspector format_inspector_;

public:
  MTLD3D12DeviceImpl(IMTLDXGIAdapter *adapter) : adapter_(adapter), command_library(adapter_->GetMTLDevice()) {}

  ~MTLD3D12DeviceImpl() {}

  HRESULT
  Initialize() {
    WMT::Reference<WMT::Error> err;
    residency_set_ = adapter_->GetMTLDevice().newResidencySet(0, err);
    if (!residency_set_) {
      ERR("Failed to create MTLResidencySet: ", err.description().getUTF8String());
      return E_FAIL;
    }
    format_inspector_.Inspect(GetMTLDevice());
    
    WMTDepthStencilInfo info{};
    info.depth_compare_function = WMTCompareFunctionAlways;
    default_depth_stencil_state = GetMTLDevice().newDepthStencilState(info);
    fence_helper = GetMTLDevice().newCommandQueue(64);
    if (!fence_helper) {
      ERR("Failed to create the fence helper queue");
      return E_FAIL;
    }

    return S_OK;
  };

  WMT::Device
  GetMTLDevice() {
    return adapter_->GetMTLDevice();
  };

  D3D_FEATURE_LEVEL
  GetFeatureLevel() {
    return D3D_FEATURE_LEVEL_11_0; // FIXME
  };

  HRESULT
  GetAdapter(REFIID riid, void **ppAdapter) {
    return adapter_->QueryInterface(riid, ppAdapter);
  };

  UINT STDMETHODCALLTYPE
  GetNodeCount() {
    DXMT_STAT_SCOPE("device.GetNodeCount");
    return 1; // FIXME
  };

  HRESULT
  STDMETHODCALLTYPE
  QueryInterface(REFIID riid, void **ppvObject) {
    DXMT_STAT_SCOPE("device.QueryInterface");
    if (ppvObject == nullptr)
      return E_POINTER;

    *ppvObject = nullptr;

    if (riid == __uuidof(IUnknown) || riid == __uuidof(ID3D12Object) || riid == __uuidof(ID3D12Device) ||
        riid == __uuidof(ID3D12Device1) || riid == __uuidof(ID3D12Device2) || riid == __uuidof(ID3D12Device3) ||
        riid == __uuidof(ID3D12Device4) || riid == __uuidof(ID3D12Device5) || riid == __uuidof(ID3D12Device6) ||
        riid == __uuidof(ID3D12Device7) || riid == __uuidof(ID3D12Device8)) {
      *ppvObject = ref(this);
      return S_OK;
    }

    if (riid == __uuidof(IDXGIDevice) || riid == __uuidof(IDXGIDevice1) || riid == __uuidof(IDXGIDevice2) ||
        riid == __uuidof(IDXGIDevice3) || riid == __uuidof(IDXGIDevice4))
      return E_NOINTERFACE;

    if (riid == __uuidof(ID3D10Device) || riid == __uuidof(ID3D10Device1))
      return E_NOINTERFACE;

    if (riid == __uuidof(ID3D11Device) || riid == __uuidof(ID3D11Device1) || riid == __uuidof(ID3D11Device2) ||
        riid == __uuidof(ID3D11Device3) || riid == __uuidof(ID3D11Device4) || riid == __uuidof(ID3D11Device5))
      return E_NOINTERFACE;

    if (riid == kD3D12DeviceDownlevelUUID)
      return E_NOINTERFACE;

    if (logQueryInterfaceError(__uuidof(ID3D12Device1), riid)) {
      WARN("D3D12Device: Unknown interface query ", str::format(riid));
    }

    return E_NOINTERFACE;
  }

  HRESULT STDMETHODCALLTYPE
  CreateCommandQueue(const D3D12_COMMAND_QUEUE_DESC *pDesc, REFIID riid, void **ppCommandQueue) {
    DXMT_STAT_SCOPE("device.CreateCommandQueue");
    if (pDesc->Flags)
      WARN("CreateCommandQueue: flags ignored: ", pDesc->Flags);
    return dxmt::CreateCommandQueue(this, pDesc, riid, ppCommandQueue);
  };

  HRESULT STDMETHODCALLTYPE
  CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE Type, REFIID riid, void **ppCommandAllocator) {
    DXMT_STAT_SCOPE("device.CreateCommandAllocator");
    return dxmt::CreateCommandAllocator(this, Type, riid, ppCommandAllocator);
  };

  HRESULT STDMETHODCALLTYPE
  CreateGraphicsPipelineState(const D3D12_GRAPHICS_PIPELINE_STATE_DESC *pDesc, REFIID riid, void **ppPipelineState) {
    DXMT_STAT_SCOPE("device.CreateGraphicsPipelineState");
    return dxmt::CreateGraphicsPipelineState(this, pDesc, riid, ppPipelineState);
  };

  HRESULT STDMETHODCALLTYPE
  CreateComputePipelineState(const D3D12_COMPUTE_PIPELINE_STATE_DESC *pDesc, REFIID riid, void **ppPipelineState) {
    DXMT_STAT_SCOPE("device.CreateComputePipelineState");
    return dxmt::CreateComputePipelineState(this, pDesc, riid, ppPipelineState);
  };

  HRESULT STDMETHODCALLTYPE
  CreateCommandList(
      UINT NodeMask, D3D12_COMMAND_LIST_TYPE Type, ID3D12CommandAllocator *pCommandAllocator,
      ID3D12PipelineState *pInitialPipelineState, REFIID riid, void **ppCommandList
  ) {
    DXMT_STAT_SCOPE("device.CreateCommandList");
    if (!pCommandAllocator)
      return E_INVALIDARG;
    auto allocator = static_cast<MTLD3D12CommandAllocator *>(pCommandAllocator);
    return allocator->CreateCommandList(NodeMask, Type, pInitialPipelineState, riid, ppCommandList);
  };

  HRESULT STDMETHODCALLTYPE
  CheckFeatureSupport(D3D12_FEATURE Feature, void *pFeatureData, UINT DataSize) {
    DXMT_STAT_SCOPE("device.CheckFeatureSupport");
    auto metal = GetMTLDevice();
    switch (Feature) {
    case D3D12_FEATURE_ARCHITECTURE: {
      if (DataSize != sizeof(D3D12_FEATURE_DATA_ARCHITECTURE))
        return E_INVALIDARG;
      auto *out = reinterpret_cast<D3D12_FEATURE_DATA_ARCHITECTURE *>(pFeatureData);
      if (out->NodeIndex > 0)
        return E_INVALIDARG;
      out->CacheCoherentUMA = FALSE;
      out->TileBasedRenderer = TRUE;
      out->UMA = !advertise_numa_;
      return S_OK;
    }
    case D3D12_FEATURE_ARCHITECTURE1: {
      if (DataSize != sizeof(D3D12_FEATURE_DATA_ARCHITECTURE1))
        return E_INVALIDARG;
      auto *out = reinterpret_cast<D3D12_FEATURE_DATA_ARCHITECTURE1 *>(pFeatureData);
      if (out->NodeIndex > 0)
        return E_INVALIDARG;
      out->CacheCoherentUMA = FALSE;
      out->TileBasedRenderer = TRUE;
      out->UMA = !advertise_numa_;
      out->IsolatedMMU = FALSE;
      return S_OK;
    }
    case D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS: {
      if (DataSize != sizeof(D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS))
        return E_INVALIDARG;
      auto *out = reinterpret_cast<D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS *>(pFeatureData);

      if (out->SampleCount == 0) {
        out->Flags = D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE;
        out->NumQualityLevels = 0;
        return E_FAIL;
      }

      if (out->Format == DXGI_FORMAT_UNKNOWN) {
        out->NumQualityLevels = out->SampleCount == 0 ? 1 : 0;
        return S_OK;
      }

      MTL_DXGI_FORMAT_DESC format_desc;
      HRESULT hr = MTLQueryDXGIFormat(metal, out->Format, format_desc);
      if (SUCCEEDED(hr) && out->SampleCount) {
        out->Flags = D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE;
        out->NumQualityLevels = metal.supportsTextureSampleCount(out->SampleCount) ? 1 : 0;
      } else {
        out->Flags = D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE;
        out->NumQualityLevels = 0;
        return E_FAIL;
      }
      return S_OK;
    }
    case D3D12_FEATURE_ROOT_SIGNATURE: {
      if (DataSize != sizeof(D3D12_FEATURE_DATA_ROOT_SIGNATURE))
        return E_INVALIDARG;
      auto *out = reinterpret_cast<D3D12_FEATURE_DATA_ROOT_SIGNATURE *>(pFeatureData);
      switch (out->HighestVersion) {
      default:
        return E_INVALIDARG;
      case D3D_ROOT_SIGNATURE_VERSION_1:
        out->HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1;
        break;
      case D3D_ROOT_SIGNATURE_VERSION_1_1:
        out->HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;
        break;
      }
      return S_OK;
    }
    case D3D12_FEATURE_FEATURE_LEVELS: {
      if (DataSize != sizeof(D3D12_FEATURE_DATA_FEATURE_LEVELS))
        return E_INVALIDARG;
      auto *out = reinterpret_cast<D3D12_FEATURE_DATA_FEATURE_LEVELS *>(pFeatureData);
      if (!out->NumFeatureLevels)
        return E_INVALIDARG;
      D3D_FEATURE_LEVEL max_level = {};
      for (unsigned i = 0; i < out->NumFeatureLevels; i++)
        max_level = std::max(out->pFeatureLevelsRequested[i], max_level);
      out->MaxSupportedFeatureLevel = std::min(max_level, SM6Caps() ? D3D_FEATURE_LEVEL_12_1 : D3D_FEATURE_LEVEL_11_1);
      return S_OK;
    }
    case D3D12_FEATURE_FORMAT_INFO:  {
       if (DataSize != sizeof(D3D12_FEATURE_DATA_FORMAT_INFO))
        return E_INVALIDARG;
      auto *out = reinterpret_cast<D3D12_FEATURE_DATA_FORMAT_INFO *>(pFeatureData);
      if (out->Format == DXGI_FORMAT_UNKNOWN) {
        out->PlaneCount = 1;
        return S_OK;
      }
      MTL_DXGI_FORMAT_DESC format_desc;
      HRESULT hr = MTLQueryDXGIFormat(metal, out->Format, format_desc);
      if (FAILED(hr))
        return E_FAIL;

      out->PlaneCount = format_desc.PlanarCount;
      return S_OK;
    }
    case D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT: {
      if (DataSize != sizeof(D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT))
        return E_INVALIDARG;
      auto *out = reinterpret_cast<D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT *>(pFeatureData);
      out->MaxGPUVirtualAddressBitsPerProcess = 48;
      out->MaxGPUVirtualAddressBitsPerResource = 48;
      return S_OK;
    }
    case D3D12_FEATURE_SHADER_MODEL: {
      if (DataSize != sizeof(D3D12_FEATURE_DATA_SHADER_MODEL))
        return E_INVALIDARG;
      auto *out = reinterpret_cast<D3D12_FEATURE_DATA_SHADER_MODEL *>(pFeatureData);
      out->HighestShaderModel =
          SM6Caps() ? std::min(out->HighestShaderModel, D3D_SHADER_MODEL_6_7) : D3D_SHADER_MODEL_5_1;
      return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS: {
      if (DataSize != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS))
        return E_INVALIDARG;
      auto *out = reinterpret_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS *>(pFeatureData);
      out->DoublePrecisionFloatShaderOps = FALSE;
      out->OutputMergerLogicOp = FALSE;
      out->MinPrecisionSupport = D3D12_SHADER_MIN_PRECISION_SUPPORT_16_BIT;
      out->TiledResourcesTier = D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED;
      out->ResourceBindingTier = SM6Caps() ? D3D12_RESOURCE_BINDING_TIER_3 : D3D12_RESOURCE_BINDING_TIER_2;
      out->PSSpecifiedStencilRefSupported = TRUE;
      out->TypedUAVLoadAdditionalFormats = TRUE;
      out->ROVsSupported = TRUE;
      out->ConservativeRasterizationTier = D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED;
      out->MaxGPUVirtualAddressBitsPerResource = 48;
      out->StandardSwizzle64KBSupported = TRUE;
      out->CrossNodeSharingTier = D3D12_CROSS_NODE_SHARING_TIER_NOT_SUPPORTED;
      out->CrossAdapterRowMajorTextureSupported = FALSE;
      out->VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation = TRUE;
      out->ResourceHeapTier = D3D12_RESOURCE_HEAP_TIER_2;
      return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS16: {
      if (DataSize != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS16))
        return E_INVALIDARG;
      auto *out = reinterpret_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS16 *>(pFeatureData);
      out->GPUUploadHeapSupported = FALSE;    // TODO(d3d12): gpu upload heap
      out->DynamicDepthBiasSupported = FALSE; // TODO(d3d12): ID3D12GraphicsCommandList9::RSSetDepthBias
      return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS2: {
      if (DataSize != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS2))
        return E_INVALIDARG;
      auto *out = reinterpret_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS2 *>(pFeatureData);
      out->DepthBoundsTestSupported = FALSE;
      out->ProgrammableSamplePositionsTier = D3D12_PROGRAMMABLE_SAMPLE_POSITIONS_TIER_NOT_SUPPORTED;
      return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS3: {
      if (DataSize != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS3))
        return E_INVALIDARG;
      auto *out = reinterpret_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS3 *>(pFeatureData);
      out->CastingFullyTypedFormatSupported = TRUE;
      out->BarycentricsSupported = FALSE;
      out->CopyQueueTimestampQueriesSupported = FALSE;
      out->ViewInstancingTier = D3D12_VIEW_INSTANCING_TIER_NOT_SUPPORTED;
      out->WriteBufferImmediateSupportFlags = D3D12_COMMAND_LIST_SUPPORT_FLAG_NONE;
      return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS1: {
      if (DataSize != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS1))
        return E_INVALIDARG;
      auto *out = reinterpret_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS1 *>(pFeatureData);
      out->WaveOps = SM6Caps();
      out->WaveLaneCountMin = SM6Caps() ? 32 : 0;
      out->WaveLaneCountMax = SM6Caps() ? 32 : 0;
      out->TotalLaneCount = 0;
      // If CheckFeatureSupport succeeds this value will always be true.
      out->ExpandedComputeResourceStates = TRUE;
      out->Int64ShaderOps = FALSE;
      return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS9: {
      if (DataSize != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS9))
        return E_INVALIDARG;
      auto *out = reinterpret_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS9 *>(pFeatureData);
      *out = {};
      out->AtomicInt64OnTypedResourceSupported = SM6Caps();
      out->AtomicInt64OnGroupSharedSupported = SM6Caps();
      return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS12: {
      if (DataSize != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS12))
        return E_INVALIDARG;
      auto *out = reinterpret_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS12 *>(pFeatureData);
      out->RelaxedFormatCastingSupported = FALSE;
      out->EnhancedBarriersSupported = FALSE;
      out->MSPrimitivesPipelineStatisticIncludesCulledPrimitives = D3D12_TRI_STATE_FALSE;
      return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS4: {
      if (DataSize != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS4))
        return E_INVALIDARG;
      auto *out = reinterpret_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS4 *>(pFeatureData);
      out->MSAA64KBAlignedTextureSupported = TRUE;
      out->Native16BitShaderOpsSupported = FALSE; // TODO(d3d12): should be true
      // TODO(d3d12): revise when d3d12 shared resource is implemented
      out->SharedResourceCompatibilityTier = D3D12_SHARED_RESOURCE_COMPATIBILITY_TIER_0;
      return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS7: {
      if (DataSize != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS7))
        return E_INVALIDARG;
      auto *out = reinterpret_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS7 *>(pFeatureData);
      out->MeshShaderTier = D3D12_MESH_SHADER_TIER_NOT_SUPPORTED;
      out->SamplerFeedbackTier = D3D12_SAMPLER_FEEDBACK_TIER_NOT_SUPPORTED;
      return S_OK;
    }
    case D3D12_FEATURE_SHADER_CACHE: {
      if (DataSize != sizeof(D3D12_FEATURE_DATA_SHADER_CACHE))
        return E_INVALIDARG;
      auto *out = reinterpret_cast<D3D12_FEATURE_DATA_SHADER_CACHE *>(pFeatureData);
      out->SupportFlags =
          D3D12_SHADER_CACHE_SUPPORT_AUTOMATIC_INPROC_CACHE | D3D12_SHADER_CACHE_SUPPORT_AUTOMATIC_DISK_CACHE;
      return S_OK;
    }
    case D3D12_FEATURE_FORMAT_SUPPORT: {
      if (DataSize != sizeof(D3D12_FEATURE_DATA_FORMAT_SUPPORT))
        return E_INVALIDARG;
      auto *out = reinterpret_cast<D3D12_FEATURE_DATA_FORMAT_SUPPORT *>(pFeatureData);

      if (out->Format == DXGI_FORMAT_UNKNOWN) {
        out->Support1 = D3D12_FORMAT_SUPPORT1_BUFFER;
        out->Support2 = {};
        return S_OK;
      }

      // TODO(d3d12): report correct support
      out->Support1 = (D3D12_FORMAT_SUPPORT1)0xffffffff;
      out->Support2 = (D3D12_FORMAT_SUPPORT2)0xffffffff;
      return S_OK;
    }
#define FEATURE_DATA(T)                                                                                                \
  if (DataSize != sizeof(T))                                                                                           \
    return E_INVALIDARG;                                                                                               \
  auto *out = reinterpret_cast<T *>(pFeatureData);                                                                     \
  *out = {};
    // Newer queries D3DMetal answers (MacNeutron): DXMT's own capabilities, none of what the fork lacks.
    case D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_SUPPORT: {
      if (DataSize != sizeof(D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_SUPPORT))
        return E_INVALIDARG;
      auto *out = reinterpret_cast<D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_SUPPORT *>(pFeatureData);
      if (out->NodeIndex)
        return E_INVALIDARG;
      out->Support = D3D12_PROTECTED_RESOURCE_SESSION_SUPPORT_FLAG_NONE;
      return S_OK;
    }
    case D3D12_FEATURE_EXISTING_HEAPS: { // OpenExistingHeapFrom* are E_NOTIMPL
      FEATURE_DATA(D3D12_FEATURE_DATA_EXISTING_HEAPS)
      return S_OK;
    }
    case D3D12_FEATURE_CROSS_NODE:
    case D3D12_FEATURE_DISPLAYABLE:
      return E_INVALIDARG; // as D3DMetal
    case D3D12_FEATURE_D3D12_OPTIONS5: { // no raytracing, render passes tier 0
      FEATURE_DATA(D3D12_FEATURE_DATA_D3D12_OPTIONS5)
      out->RenderPassesTier = D3D12_RENDER_PASS_TIER_0;
      out->RaytracingTier = D3D12_RAYTRACING_TIER_NOT_SUPPORTED;
      return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS6: { // no variable-rate shading
      FEATURE_DATA(D3D12_FEATURE_DATA_D3D12_OPTIONS6)
      return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS8: {
      FEATURE_DATA(D3D12_FEATURE_DATA_D3D12_OPTIONS8)
      return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS10: {
      FEATURE_DATA(D3D12_FEATURE_DATA_D3D12_OPTIONS10)
      return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS11: {
      FEATURE_DATA(D3D12_FEATURE_DATA_D3D12_OPTIONS11)
      return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS13: {
      FEATURE_DATA(D3D12_FEATURE_DATA_D3D12_OPTIONS13)
      return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS14: {
      FEATURE_DATA(D3D12_FEATURE_DATA_D3D12_OPTIONS14)
      return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS15: {
      FEATURE_DATA(D3D12_FEATURE_DATA_D3D12_OPTIONS15)
      return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS17: {
      FEATURE_DATA(D3D12_FEATURE_DATA_D3D12_OPTIONS17)
      return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS18: {
      FEATURE_DATA(D3D12_FEATURE_DATA_D3D12_OPTIONS18)
      return S_OK;
    }
    case kFeatureOptions19: {
      FEATURE_DATA(D3D12_FEATURE_DATA_D3D12_OPTIONS19_MN)
      out->MaxSamplerDescriptorHeapSize = 2048;
      out->MaxSamplerDescriptorHeapSizeWithStaticSamplers = 2048;
      out->MaxViewDescriptorHeapSize = 1000000;
      return S_OK;
    }
    case kFeatureOptions21: {
      FEATURE_DATA(D3D12_FEATURE_DATA_D3D12_OPTIONS21_MN)
      out->ExecuteIndirectTier = 10; // D3D12_EXECUTE_INDIRECT_TIER_1_0
      return S_OK;
    }
#undef FEATURE_DATA
    default:
      break;
    }
    ERR("CheckFeatureSupport: unhandled feature ", Feature);
    return E_NOTIMPL;
  };

  HRESULT STDMETHODCALLTYPE
  CreateDescriptorHeap(const D3D12_DESCRIPTOR_HEAP_DESC *pDesc, REFIID riid, void **ppDescriptorHeap) {
    DXMT_STAT_SCOPE("device.CreateDescriptorHeap");
    return dxmt::CreateDescriptorHeap(this, pDesc, riid, ppDescriptorHeap);
  };

  UINT STDMETHODCALLTYPE
  GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE DescriptorHeapType) {
    DXMT_STAT_SCOPE("device.GetDescriptorHandleIncrementSize");
    switch (DescriptorHeapType) {
    case D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV:
    case D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER:
    case D3D12_DESCRIPTOR_HEAP_TYPE_RTV:
    case D3D12_DESCRIPTOR_HEAP_TYPE_DSV:
      return 32;
    default:
      break;
    }
    return 0;
  };

  HRESULT STDMETHODCALLTYPE
  CreateRootSignature(
      UINT NodeMask, const void *pBytecode, SIZE_T BytecodeLength, REFIID riid, void **ppRootSignature
  ) {
    DXMT_STAT_SCOPE("device.CreateRootSignature");
    return dxmt::CreateRootSignature(this, NodeMask, pBytecode, BytecodeLength, riid, ppRootSignature);
  };

  void STDMETHODCALLTYPE
  CreateConstantBufferView(const D3D12_CONSTANT_BUFFER_VIEW_DESC *pDesc, D3D12_CPU_DESCRIPTOR_HANDLE Descriptor) {
    DXMT_STAT_SCOPE("device.CreateConstantBufferView");
    auto [Heap, Index] = GetShaderVisibleDescriptorHeap(this, Descriptor);
    if (pDesc)
      Heap->AddConstantBufferView(Index, pDesc->BufferLocation, pDesc->SizeInBytes);
    else
      Heap->AddConstantBufferView(Index, 0, 0);
  };

  void STDMETHODCALLTYPE
  CreateShaderResourceView(
      ID3D12Resource *pResource, const D3D12_SHADER_RESOURCE_VIEW_DESC *pDesc, D3D12_CPU_DESCRIPTOR_HANDLE Descriptor
  ) {
    DXMT_STAT_SCOPE("device.CreateShaderResourceView");
    if (!pResource) {
      auto [Heap, Index] = GetShaderVisibleDescriptorHeap(this, Descriptor);
      Heap->AddShaderResourceView(Index, pDesc);
      return;
    }
    auto d3d12res = static_cast<MTLD3D12Resource *>(pResource);
    d3d12res->CreateShaderResourceView(pDesc, Descriptor);
  };

  void STDMETHODCALLTYPE
  CreateUnorderedAccessView(
      ID3D12Resource *pResource, ID3D12Resource *pCounter, const D3D12_UNORDERED_ACCESS_VIEW_DESC *pDesc,
      D3D12_CPU_DESCRIPTOR_HANDLE Descriptor
  ) {
    DXMT_STAT_SCOPE("device.CreateUnorderedAccessView");
    if (!pResource) {
      auto [Heap, Index] = GetShaderVisibleDescriptorHeap(this, Descriptor);
      Heap->AddUnorderedAccessView(Index, pDesc);
      return;
    }
    auto d3d12res = static_cast<MTLD3D12Resource *>(pResource);
    d3d12res->CreateUnorderedAccessView(pCounter, pDesc, Descriptor);
  };

  void STDMETHODCALLTYPE
  CreateRenderTargetView(
      ID3D12Resource *pResource, const D3D12_RENDER_TARGET_VIEW_DESC *pDesc, D3D12_CPU_DESCRIPTOR_HANDLE Descriptor
  ) {
    DXMT_STAT_SCOPE("device.CreateRenderTargetView");
    if (!pResource) {
      auto [Heap, Index] = GetRenderTargetHeap(this, Descriptor);
      Heap->AddRenderTarget(Index, nullptr);
      return;
    }
    auto d3d12res = static_cast<MTLD3D12Resource *>(pResource);
    d3d12res->CreateRenderTargetView(pDesc, Descriptor);
  };

  void STDMETHODCALLTYPE
  CreateDepthStencilView(
      ID3D12Resource *pResource, const D3D12_DEPTH_STENCIL_VIEW_DESC *pDesc, D3D12_CPU_DESCRIPTOR_HANDLE Descriptor
  ) {
    DXMT_STAT_SCOPE("device.CreateDepthStencilView");
    if (!pResource) {
      auto [Heap, Index] = GetRenderTargetHeap(this, Descriptor);
      Heap->AddRenderTarget(Index, nullptr);
      return;
    }
    auto d3d12res = static_cast<MTLD3D12Resource *>(pResource);
    d3d12res->CreateDepthStencilView(pDesc, Descriptor);
  };

  void STDMETHODCALLTYPE
  CreateSampler(const D3D12_SAMPLER_DESC *pDesc, D3D12_CPU_DESCRIPTOR_HANDLE Descriptor) {
    DXMT_STAT_SCOPE("device.CreateSampler");
    auto [Heap, Index] = GetSamplerDescriptorHeap(this, Descriptor);
    Heap->AddSampler(Index, pDesc);
  };

  void STDMETHODCALLTYPE
  CopyDescriptors(
      UINT DstDescriptorRangeCount, const D3D12_CPU_DESCRIPTOR_HANDLE *DstDescriptorRangeOffsets,
      const UINT *DstDescriptorRangeSizes, UINT SrcDescriptorRangeCount,
      const D3D12_CPU_DESCRIPTOR_HANDLE *SrcDescriptorRangeOffsets, const UINT *SrcDescriptorRangeSizes,
      D3D12_DESCRIPTOR_HEAP_TYPE DescriptorHeapType
  ) {
    DXMT_STAT_SCOPE("device.CopyDescriptors");
    unsigned int dst_range_idx, dst_idx, src_range_idx, src_idx;
    unsigned int dst_range_size, src_range_size, copy_count;

    dst_range_idx = dst_idx = 0;
    src_range_idx = src_idx = 0;
    while (dst_range_idx < DstDescriptorRangeCount && src_range_idx < SrcDescriptorRangeCount) {
      dst_range_size = DstDescriptorRangeSizes ? DstDescriptorRangeSizes[dst_range_idx] : 1;
      src_range_size = SrcDescriptorRangeSizes ? SrcDescriptorRangeSizes[src_range_idx] : 1;

      copy_count = std::min(dst_range_size - dst_idx, src_range_size - src_idx);

      switch (DescriptorHeapType) {
      case D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV: {
        auto [DstRangeHeap, DstRangeIndex] =
            GetShaderVisibleDescriptorHeap(this, DstDescriptorRangeOffsets[dst_range_idx]);
        auto [SrcRangeHeap, SrcRangeIndex] =
            GetShaderVisibleDescriptorHeap(this, SrcDescriptorRangeOffsets[src_range_idx]);
        SrcRangeHeap->CopyDescriptors(SrcRangeIndex + src_idx, DstRangeHeap, DstRangeIndex + dst_idx, copy_count);
        break;
      }

      case D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER: {
        auto [DstRangeHeap, DstRangeIndex] = GetSamplerDescriptorHeap(this, DstDescriptorRangeOffsets[dst_range_idx]);
        auto [SrcRangeHeap, SrcRangeIndex] = GetSamplerDescriptorHeap(this, SrcDescriptorRangeOffsets[src_range_idx]);
        SrcRangeHeap->CopyDescriptors(SrcRangeIndex + src_idx, DstRangeHeap, DstRangeIndex + dst_idx, copy_count);
        break;
      }
      case D3D12_DESCRIPTOR_HEAP_TYPE_RTV:
      case D3D12_DESCRIPTOR_HEAP_TYPE_DSV: {
        auto [DstRangeHeap, DstRangeIndex] = GetRenderTargetHeap(this, DstDescriptorRangeOffsets[dst_range_idx]);
        auto [SrcRangeHeap, SrcRangeIndex] = GetRenderTargetHeap(this, SrcDescriptorRangeOffsets[src_range_idx]);
        SrcRangeHeap->CopyDescriptors(SrcRangeIndex + src_idx, DstRangeHeap, DstRangeIndex + dst_idx, copy_count);
        break;
      }
      default:
        return;
      }

      dst_idx += copy_count;
      src_idx += copy_count;

      if (dst_idx >= dst_range_size) {
        ++dst_range_idx;
        dst_idx = 0;
      }
      if (src_idx >= src_range_size) {
        ++src_range_idx;
        src_idx = 0;
      }
    }
  };

  void STDMETHODCALLTYPE
  CopyDescriptorsSimple(
      UINT DescriptorCount, const D3D12_CPU_DESCRIPTOR_HANDLE DstDescriptorRangeOffset,
      const D3D12_CPU_DESCRIPTOR_HANDLE SrcDescriptorRangeOffset, D3D12_DESCRIPTOR_HEAP_TYPE DescriptorHeapType
  ) {
    DXMT_STAT_SCOPE("device.CopyDescriptorsSimple");
    CopyDescriptors(
        1, &DstDescriptorRangeOffset, &DescriptorCount, 1, &SrcDescriptorRangeOffset, &DescriptorCount,
        DescriptorHeapType
    );
  };

  D3D12_RESOURCE_ALLOCATION_INFO *STDMETHODCALLTYPE
  GetResourceAllocationInfo(
      D3D12_RESOURCE_ALLOCATION_INFO *__ret, UINT VisibleMask, UINT ResourceDestCount, const D3D12_RESOURCE_DESC *pDescs
  ) {
    DXMT_STAT_SCOPE("device.GetResourceAllocationInfo");
    return GetResourceAllocationInfo1(__ret, VisibleMask, ResourceDestCount, pDescs, nullptr);
  };

  D3D12_HEAP_PROPERTIES *STDMETHODCALLTYPE
  GetCustomHeapProperties(D3D12_HEAP_PROPERTIES *__ret, UINT NodeMask, D3D12_HEAP_TYPE HeapType) {
    DXMT_STAT_SCOPE("device.GetCustomHeapProperties");
    __ret->Type = D3D12_HEAP_TYPE_CUSTOM;
    __ret->CreationNodeMask = 1;
    __ret->VisibleNodeMask = 1;
    switch (HeapType) {
    case D3D12_HEAP_TYPE_DEFAULT:
      __ret->CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE;
      __ret->MemoryPoolPreference = advertise_numa_ ? D3D12_MEMORY_POOL_L1 : D3D12_MEMORY_POOL_L0;
      break;
    case D3D12_HEAP_TYPE_UPLOAD:
      __ret->CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE;
      __ret->MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
      break;
    case D3D12_HEAP_TYPE_READBACK:
      __ret->CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
      __ret->MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
      break;
    default:
      E_INVALIDARG;
    }

    return __ret;
  };

  HRESULT STDMETHODCALLTYPE
  CreateCommittedResource(
      const D3D12_HEAP_PROPERTIES *pHeapProps, D3D12_HEAP_FLAGS HeapFlags, const D3D12_RESOURCE_DESC *pDesc,
      D3D12_RESOURCE_STATES InitialState, const D3D12_CLEAR_VALUE *OptimizedClearValue, REFIID riid, void **ppResource
  ) {
    DXMT_STAT_SCOPE("device.CreateCommittedResource");
    InitReturnPtr(ppResource);
    HRESULT hr = S_OK;
    hr = ValidateHeapProperties(pHeapProps, HeapFlags, advertise_numa_);
    if (FAILED(hr))
      return hr;
    hr = ValidateResourceDescs(pDesc, pHeapProps);
    if (FAILED(hr))
      return hr;
    hr = ValidateResourceStates(InitialState, pHeapProps);
    if (FAILED(hr))
      return hr;
    switch (pDesc->Dimension) {
    case D3D12_RESOURCE_DIMENSION_TEXTURE1D:
    case D3D12_RESOURCE_DIMENSION_TEXTURE2D:
    case D3D12_RESOURCE_DIMENSION_TEXTURE3D:
      return CreateCommittedTexture(
          this, pHeapProps, HeapFlags, pDesc, InitialState, OptimizedClearValue, riid, ppResource
      );
    case D3D12_RESOURCE_DIMENSION_BUFFER:
      return CreateCommittedBuffer(
          this, pHeapProps, HeapFlags, pDesc, InitialState, OptimizedClearValue, riid, ppResource
      );
    default:
      break;
    }
    return E_INVALIDARG;
  };

  HRESULT STDMETHODCALLTYPE
  CreateHeap(const D3D12_HEAP_DESC *pDesc, REFIID riid, void **ppHeap) {
    DXMT_STAT_SCOPE("device.CreateHeap");
    HRESULT hr = S_OK;
    hr = ValidateHeapProperties(&pDesc->Properties, pDesc->Flags, advertise_numa_);
    if (FAILED(hr))
      return hr;
    return dxmt::CreateHeap(this, pDesc, riid, ppHeap);
  };

  HRESULT STDMETHODCALLTYPE
  CreatePlacedResource(
      ID3D12Heap *pHeap, UINT64 Offset, const D3D12_RESOURCE_DESC *pDesc, D3D12_RESOURCE_STATES InitialState,
      const D3D12_CLEAR_VALUE *OptimizedClearValue, REFIID riid, void **ppResource
  ) {
    DXMT_STAT_SCOPE("device.CreatePlacedResource");
    InitReturnPtr(ppResource);
    if (!pHeap)
      return E_INVALIDARG;
    auto d3d12heap = static_cast<MTLD3D12Heap *>(pHeap);
    auto heap_desc = d3d12heap->GetDesc();
    HRESULT hr = S_OK;
    hr = ValidateHeapProperties(&heap_desc.Properties, heap_desc.Flags, advertise_numa_);
    if (FAILED(hr))
      return hr;
    hr = ValidateResourceDescs(pDesc, &heap_desc.Properties);
    if (FAILED(hr))
      return hr;
    hr = ValidateResourceStates(InitialState, &heap_desc.Properties);
    if (FAILED(hr))
      return hr;
    switch (pDesc->Dimension) {
    case D3D12_RESOURCE_DIMENSION_TEXTURE1D:
    case D3D12_RESOURCE_DIMENSION_TEXTURE2D:
    case D3D12_RESOURCE_DIMENSION_TEXTURE3D:
      return CreatePlacedTexture(this, d3d12heap, Offset, pDesc, InitialState, OptimizedClearValue, riid, ppResource);
    case D3D12_RESOURCE_DIMENSION_BUFFER:
      return CreatePlacedBuffer(this, d3d12heap, Offset, pDesc, InitialState, OptimizedClearValue, riid, ppResource);
    default:
      break;
    }
    return E_INVALIDARG;
  };

  HRESULT STDMETHODCALLTYPE
  CreateReservedResource(
      const D3D12_RESOURCE_DESC *pDesc, D3D12_RESOURCE_STATES InitialState,
      const D3D12_CLEAR_VALUE *OptimizedClearValue, REFIID riid, void **resource
  ) {
    DXMT_STAT_SCOPE("device.CreateReservedResource");
    return E_NOTIMPL;
  };

  HRESULT STDMETHODCALLTYPE
  CreateSharedHandle(
      ID3D12DeviceChild *object, const SECURITY_ATTRIBUTES *attributes, DWORD access, const WCHAR *name, HANDLE *handle
  ) {
    DXMT_STAT_SCOPE("device.CreateSharedHandle");
    return E_NOTIMPL;
  };

  HRESULT STDMETHODCALLTYPE
  OpenSharedHandle(HANDLE handle, REFIID riid, void **object) {
    DXMT_STAT_SCOPE("device.OpenSharedHandle");
    return E_NOTIMPL;
  };

  HRESULT STDMETHODCALLTYPE
  OpenSharedHandleByName(const WCHAR *name, DWORD access, HANDLE *handle) {
    DXMT_STAT_SCOPE("device.OpenSharedHandleByName");
    return E_NOTIMPL;
  };

  HRESULT STDMETHODCALLTYPE
  MakeResident(UINT ObjectCount, ID3D12Pageable *const *objects) {
    DXMT_STAT_SCOPE("device.MakeResident");
    return S_OK;
  };

  // Unified memory never pages a resource out, and DXMT's residency sets keep heaps resident: residency calls do
  // nothing, as on D3DMetal.
  HRESULT STDMETHODCALLTYPE
  Evict(UINT ObjectCount, ID3D12Pageable *const *objects) {
    DXMT_STAT_SCOPE("device.Evict");
    return S_OK;
  };

  HRESULT STDMETHODCALLTYPE
  CreateFence(UINT64 InitialValue, D3D12_FENCE_FLAGS Flags, REFIID riid, void **ppFence) {
    DXMT_STAT_SCOPE("device.CreateFence");
    return dxmt::CreateFence(this, InitialValue, Flags, riid, ppFence);
  };

  HRESULT STDMETHODCALLTYPE
  GetDeviceRemovedReason() {
    DXMT_STAT_SCOPE("device.GetDeviceRemovedReason");
    return S_OK;
  };

  void STDMETHODCALLTYPE GetCopyableFootprints(
      const D3D12_RESOURCE_DESC *pDesc, UINT FirstSubresource, UINT SubresourceCount, UINT64 BaseOffset,
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT *pLayouts, UINT *pNumRows, UINT64 *pRowSizeInBytes, UINT64 *pTotalBytes
  ) {
    DXMT_STAT_SCOPE("device.GetCopyableFootprints");
    UINT64 TotalBytes = 0;
    UINT64 Offset = 0;
    UINT BlockWidth = 1;
    do {
      if (!pDesc)
        break;

      UINT PlaneCount = 1;
      UINT PerPlaneSubresources = DecomposeSubresource(*pDesc, 0);
      DXGI_FORMAT PlaneFormats[2] = {};
      UINT PlaneBytesPerTexel[2] = {};

      MTL_DXGI_FORMAT_DESC FormatDesc;

      if (pDesc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
        if (pDesc->Format != DXGI_FORMAT_UNKNOWN)
          break;
        PlaneFormats[0] = DXGI_FORMAT_UNKNOWN;
        PlaneBytesPerTexel[0] = 1;
      } else {
        if (FAILED(MTLQueryDXGIFormat(GetMTLDevice(), pDesc->Format, FormatDesc)))
          break;

        if (pDesc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D && pDesc->Height != 1)
          break;

        if (FormatDesc.Flag & MTL_DXGI_FORMAT_BC)
          BlockWidth = 4;

        if (FormatDesc.PlanarCount > 1) {
          assert(FormatDesc.Flag & (MTL_DXGI_FORMAT_DEPTH_PLANER | MTL_DXGI_FORMAT_STENCIL_PLANER));
          PlaneCount = FormatDesc.PlanarCount;
          PlaneFormats[0] = DXGI_FORMAT_R32_TYPELESS;
          PlaneBytesPerTexel[0] = 4;
          PlaneFormats[1] = DXGI_FORMAT_R8_TYPELESS;
          PlaneBytesPerTexel[1] = 1;
        } else {
          PlaneFormats[0] = pDesc->Format;
          PlaneBytesPerTexel[0] = FormatDesc.BytesPerTexel;
          if (PlaneBytesPerTexel[0] == 0)
            IMPLEMENT_ME
        }
      }

      if (pDesc->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D)
        PerPlaneSubresources *= pDesc->DepthOrArraySize;

      if (FirstSubresource >= PerPlaneSubresources * PlaneCount ||
          SubresourceCount > PerPlaneSubresources * PlaneCount - FirstSubresource) {
        WARN("GetCopyableFootprints: subresource is out of range");
        break;
      }

      for (unsigned i = 0; i < SubresourceCount; i++) {
        auto Subresource = FirstSubresource + i;
        auto Plane = 0u, MipLevel = 0u;
        DecomposeSubresource(*pDesc, Subresource, &MipLevel, NULL, &Plane);
        auto Extent = GetResourceExtent(*pDesc, MipLevel);
        auto Width = align(Extent.right, BlockWidth);
        auto Height = align(Extent.bottom, BlockWidth);
        auto RowCount = Height / BlockWidth;
        auto Depth = Extent.back;
        auto RowSize = (Width / BlockWidth) * PlaneBytesPerTexel[Plane];
        auto RowPitch = align(RowSize, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT * PlaneCount);
        if (pLayouts) {
          pLayouts[i].Offset = BaseOffset + Offset;
          pLayouts[i].Footprint.Format = PlaneFormats[Plane];
          pLayouts[i].Footprint.Width = Width;
          pLayouts[i].Footprint.Height = Height;
          pLayouts[i].Footprint.Depth = Depth;
          pLayouts[i].Footprint.RowPitch = RowPitch;
        }
        if (pNumRows)
          pNumRows[i] = RowCount;
        if (pRowSizeInBytes)
          pRowSizeInBytes[i] = RowSize;

        auto SubresourceSize = RowPitch * (RowCount - 1) + RowSize;
        SubresourceSize =
            align(SubresourceSize, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT * PlaneCount) * (Depth - 1) + SubresourceSize;

        TotalBytes = Offset + SubresourceSize;
        Offset = align(TotalBytes, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
      }
      if (pTotalBytes)
        *pTotalBytes = TotalBytes;
      return;
    } while (0);
    for (unsigned i = 0; i < SubresourceCount; i++) {
      if (pLayouts) {
        pLayouts[i].Offset = ~0ull;
        pLayouts[i].Footprint.Format = ~(DXGI_FORMAT)0u;
        pLayouts[i].Footprint.Width = ~0u;
        pLayouts[i].Footprint.Height = ~0u;
        pLayouts[i].Footprint.Depth = ~0u;
        pLayouts[i].Footprint.RowPitch = ~0u;
      }
      if (pNumRows)
        pNumRows[i] = ~0u;
      if (pRowSizeInBytes)
        pRowSizeInBytes[i] = ~0ull;
    }
    if (pTotalBytes)
      *pTotalBytes = UINT64_MAX;
  };

  HRESULT STDMETHODCALLTYPE
  CreateQueryHeap(const D3D12_QUERY_HEAP_DESC *pDesc, REFIID riid, void **ppHeap) {
    DXMT_STAT_SCOPE("device.CreateQueryHeap");
    return dxmt::CreateQueryHeap(this, pDesc, riid, ppHeap);
  };

  HRESULT STDMETHODCALLTYPE
  SetStablePowerState(WINBOOL Enable) {
    DXMT_STAT_SCOPE("device.SetStablePowerState");
    return E_NOTIMPL;
  };

  HRESULT STDMETHODCALLTYPE
  CreateCommandSignature(
      const D3D12_COMMAND_SIGNATURE_DESC *pDesc, ID3D12RootSignature *pRootSignature, REFIID riid,
      void **ppCommandSignature
  ) {
    DXMT_STAT_SCOPE("device.CreateCommandSignature");
    return dxmt::CreateCommandSignature(this, pDesc, pRootSignature, riid, ppCommandSignature);
  };

  void STDMETHODCALLTYPE GetResourceTiling(
      ID3D12Resource *pResource, UINT *TotalTileCount, D3D12_PACKED_MIP_INFO *PackedMipInfo,
      D3D12_TILE_SHAPE *StandardTileShape, UINT *SubresourceTilingCount, UINT FirstSubresourceTiling,
      D3D12_SUBRESOURCE_TILING *SubresourceTilings
  ) {
    DXMT_STAT_SCOPE("device.GetResourceTiling");
    IMPLEMENT_ME
  };

  LUID *STDMETHODCALLTYPE
  GetAdapterLuid(LUID *ret) {
    DXMT_STAT_SCOPE("device.GetAdapterLuid");
    *ret = std::bit_cast<LUID>(__builtin_bswap64(adapter_->GetMTLDevice().registryID()));
    return ret;
  }

  HRESULT STDMETHODCALLTYPE
  CreatePipelineLibrary(const void *blob, SIZE_T blob_size, REFIID iid, void **lib) {
    DXMT_STAT_SCOPE("device.CreatePipelineLibrary");
    return dxmt::CreatePipelineLibrary(this, blob, blob_size, iid, lib);
  };

  HRESULT STDMETHODCALLTYPE
  SetEventOnMultipleFenceCompletion(
      ID3D12Fence *const *pFences, const UINT64 *pValues, UINT FenceCount, D3D12_MULTIPLE_FENCE_WAIT_FLAGS Flags,
      HANDLE hEvent
  ) {
    DXMT_STAT_SCOPE("device.SetEventOnMultipleFenceCompletion");
    if (!FenceCount)
      return S_OK; // nothing to wait for; D3DMetal leaves the event alone too
    if (!pFences || !pValues)
      return E_INVALIDARG;
    std::vector<Fence const *> fences(FenceCount);
    for (UINT i = 0; i < FenceCount; i++)
      fences[i] = static_cast<MTLD3D12Fence *>(pFences[i])->fence.ptr();
    bool all = Flags == D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL;
    // No event: wait here, as SetEventOnCompletion does.
    HANDLE event = hEvent ? hEvent : CreateEventW(nullptr, FALSE, FALSE, nullptr);
    event_listener.setEventOnValues(fences.data(), pValues, FenceCount, all, event);
    if (!hEvent) {
      WaitForSingleObject(event, INFINITE);
      CloseHandle(event);
    }
    return S_OK;
  };

  HRESULT STDMETHODCALLTYPE
  SetResidencyPriority(UINT ObjectCount, ID3D12Pageable *const *pObjects, const D3D12_RESIDENCY_PRIORITY *pPriorities) {
    DXMT_STAT_SCOPE("device.SetResidencyPriority");
    return S_OK;
  };

  HRESULT STDMETHODCALLTYPE
  CreatePipelineState(const D3D12_PIPELINE_STATE_STREAM_DESC *pDesc, REFIID riid, void **ppPipelineState) {
    DXMT_STAT_SCOPE("device.CreatePipelineState");
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc_cs{};
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc_graphics{};
    bool compute = false;
    HRESULT hr = ParsePipelineStream(pDesc, desc_graphics, desc_cs, compute);
    if (FAILED(hr))
      return hr;
    return compute ? CreateComputePipelineState(&desc_cs, riid, ppPipelineState)
                   : CreateGraphicsPipelineState(&desc_graphics, riid, ppPipelineState);
  }

  HRESULT STDMETHODCALLTYPE
  OpenExistingHeapFromAddress(const void *pAddress, REFIID riid, void **ppHeap) {
    DXMT_STAT_SCOPE("device.OpenExistingHeapFromAddress");
    return E_NOTIMPL;
  }

  HRESULT STDMETHODCALLTYPE
  OpenExistingHeapFromFileMapping(HANDLE hFileMapping, REFIID riid, void **ppHeap) {
    DXMT_STAT_SCOPE("device.OpenExistingHeapFromFileMapping");
    return E_NOTIMPL;
  }

  HRESULT STDMETHODCALLTYPE
  EnqueueMakeResident(
      D3D12_RESIDENCY_FLAGS Flags, UINT NumObjects, ID3D12Pageable *const *ppObjects, ID3D12Fence *pFence,
      UINT64 FenceValue
  ) {
    DXMT_STAT_SCOPE("device.EnqueueMakeResident");
    if (!pFence)
      return E_INVALIDARG;
    return pFence->Signal(FenceValue); // already resident: the fence reaches the value at once
  }

  HRESULT STDMETHODCALLTYPE
  CreateCommandList1(
      UINT NodeMask, D3D12_COMMAND_LIST_TYPE Type, D3D12_COMMAND_LIST_FLAGS Flags, REFIID riid, void **ppCommandList
  ) {
    DXMT_STAT_SCOPE("device.CreateCommandList1");
    InitReturnPtr(ppCommandList);
    if (Type != D3D12_COMMAND_LIST_TYPE_DIRECT && Type != D3D12_COMMAND_LIST_TYPE_COMPUTE &&
        Type != D3D12_COMMAND_LIST_TYPE_COPY)
      return E_INVALIDARG;
    return CreateClosedCommandList(this, riid, ppCommandList);
  }

  HRESULT STDMETHODCALLTYPE
  CreateProtectedResourceSession(const D3D12_PROTECTED_RESOURCE_SESSION_DESC *pDesc, REFIID riid, void **ppSession) {
    DXMT_STAT_SCOPE("device.CreateProtectedResourceSession");
    return E_NOTIMPL;
  }

  HRESULT STDMETHODCALLTYPE
  CreateCommittedResource1(
      const D3D12_HEAP_PROPERTIES *pHeapProps, D3D12_HEAP_FLAGS HeapFlags, const D3D12_RESOURCE_DESC *pDesc,
      D3D12_RESOURCE_STATES InitialState, const D3D12_CLEAR_VALUE *OptimizedClearValue,
      ID3D12ProtectedResourceSession *pSession, REFIID riid, void **ppResource
  ) {
    DXMT_STAT_SCOPE("device.CreateCommittedResource1");
    if (pSession) // protected sessions aren't supported (CreateProtectedResourceSession fails)
      return E_NOTIMPL;
    return CreateCommittedResource(pHeapProps, HeapFlags, pDesc, InitialState, OptimizedClearValue, riid, ppResource);
  }

  // MacNeutron: ID3D12Device5-8. Raytracing, meta commands, lifetime tracking, protected sessions and sampler feedback
  // aren't supported (the features report none of them); the D3D12_RESOURCE_DESC1 methods take the plain desc
  // (SamplerFeedbackMipRegion only matters to sampler feedback).
  static D3D12_RESOURCE_DESC
  PlainDesc(const D3D12_RESOURCE_DESC1 &d) {
    return {d.Dimension, d.Alignment, d.Width, d.Height, d.DepthOrArraySize, d.MipLevels, d.Format, d.SampleDesc, d.Layout, d.Flags};
  }

  HRESULT STDMETHODCALLTYPE
  CreateLifetimeTracker(ID3D12LifetimeOwner *owner, REFIID riid, void **tracker) {
    DXMT_STAT_SCOPE("device.CreateLifetimeTracker");
    InitReturnPtr(tracker);
    return E_NOTIMPL;
  }

  void STDMETHODCALLTYPE
  RemoveDevice() {
    DXMT_STAT_SCOPE("device.RemoveDevice");}

  HRESULT STDMETHODCALLTYPE
  EnumerateMetaCommands(UINT *count, D3D12_META_COMMAND_DESC *descs) {
    DXMT_STAT_SCOPE("device.EnumerateMetaCommands");
    if (!count)
      return E_INVALIDARG;
    *count = 0;
    return S_OK;
  }

  HRESULT STDMETHODCALLTYPE
  EnumerateMetaCommandParameters(
      REFGUID command_id, D3D12_META_COMMAND_PARAMETER_STAGE stage, UINT *total_size, UINT *count,
      D3D12_META_COMMAND_PARAMETER_DESC *descs
  ) {
    DXMT_STAT_SCOPE("device.EnumerateMetaCommandParameters");
    return E_INVALIDARG; // there are no meta commands
  }

  HRESULT STDMETHODCALLTYPE
  CreateMetaCommand(REFGUID command_id, UINT node_mask, const void *data, SIZE_T size, REFIID riid, void **command) {
    DXMT_STAT_SCOPE("device.CreateMetaCommand");
    InitReturnPtr(command);
    return E_INVALIDARG;
  }

  HRESULT STDMETHODCALLTYPE
  CreateStateObject(const D3D12_STATE_OBJECT_DESC *desc, REFIID riid, void **state_object) {
    DXMT_STAT_SCOPE("device.CreateStateObject");
    InitReturnPtr(state_object);
    return E_NOTIMPL;
  }

  void STDMETHODCALLTYPE
  GetRaytracingAccelerationStructurePrebuildInfo(
      const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS *desc,
      D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO *info
  ) {
    DXMT_STAT_SCOPE("device.GetRaytracingAccelerationStructurePrebuildInfo");
    if (info)
      *info = {};
  }

  D3D12_DRIVER_MATCHING_IDENTIFIER_STATUS STDMETHODCALLTYPE
  CheckDriverMatchingIdentifier(D3D12_SERIALIZED_DATA_TYPE type, const D3D12_SERIALIZED_DATA_DRIVER_MATCHING_IDENTIFIER *identifier) {
    DXMT_STAT_SCOPE("device.CheckDriverMatchingIdentifier");
    return D3D12_DRIVER_MATCHING_IDENTIFIER_UNSUPPORTED_TYPE;
  }

  HRESULT STDMETHODCALLTYPE
  SetBackgroundProcessingMode(
      D3D12_BACKGROUND_PROCESSING_MODE mode, D3D12_MEASUREMENTS_ACTION action, HANDLE event, WINBOOL *further_measurements
  ) {
    DXMT_STAT_SCOPE("device.SetBackgroundProcessingMode");
    if (further_measurements)
      *further_measurements = FALSE;
    return S_OK;
  }

  HRESULT STDMETHODCALLTYPE
  AddToStateObject(const D3D12_STATE_OBJECT_DESC *addition, ID3D12StateObject *grow_from, REFIID riid, void **new_state_object) {
    DXMT_STAT_SCOPE("device.AddToStateObject");
    InitReturnPtr(new_state_object);
    return E_NOTIMPL;
  }

  HRESULT STDMETHODCALLTYPE
  CreateProtectedResourceSession1(const D3D12_PROTECTED_RESOURCE_SESSION_DESC1 *desc, REFIID riid, void **session) {
    DXMT_STAT_SCOPE("device.CreateProtectedResourceSession1");
    InitReturnPtr(session);
    return E_NOTIMPL;
  }

  D3D12_RESOURCE_ALLOCATION_INFO *STDMETHODCALLTYPE
  GetResourceAllocationInfo2(
      D3D12_RESOURCE_ALLOCATION_INFO *__ret, UINT VisibleMask, UINT ResourceDescCount, const D3D12_RESOURCE_DESC1 *pDescs,
      D3D12_RESOURCE_ALLOCATION_INFO1 *pAllocationInfos
  ) {
    DXMT_STAT_SCOPE("device.GetResourceAllocationInfo2");
    std::vector<D3D12_RESOURCE_DESC> descs(ResourceDescCount);
    for (UINT i = 0; i < ResourceDescCount; i++)
      descs[i] = PlainDesc(pDescs[i]);
    return GetResourceAllocationInfo1(__ret, VisibleMask, ResourceDescCount, descs.data(), pAllocationInfos);
  }

  HRESULT STDMETHODCALLTYPE
  CreateCommittedResource2(
      const D3D12_HEAP_PROPERTIES *pHeapProps, D3D12_HEAP_FLAGS HeapFlags, const D3D12_RESOURCE_DESC1 *pDesc,
      D3D12_RESOURCE_STATES InitialState, const D3D12_CLEAR_VALUE *OptimizedClearValue,
      ID3D12ProtectedResourceSession *pSession, REFIID riid, void **ppResource
  ) {
    DXMT_STAT_SCOPE("device.CreateCommittedResource2");
    if (!pDesc)
      return E_INVALIDARG;
    auto desc = PlainDesc(*pDesc);
    return CreateCommittedResource1(pHeapProps, HeapFlags, &desc, InitialState, OptimizedClearValue, pSession, riid, ppResource);
  }

  HRESULT STDMETHODCALLTYPE
  CreatePlacedResource1(
      ID3D12Heap *pHeap, UINT64 Offset, const D3D12_RESOURCE_DESC1 *pDesc, D3D12_RESOURCE_STATES InitialState,
      const D3D12_CLEAR_VALUE *OptimizedClearValue, REFIID riid, void **ppResource
  ) {
    DXMT_STAT_SCOPE("device.CreatePlacedResource1");
    if (!pDesc)
      return E_INVALIDARG;
    auto desc = PlainDesc(*pDesc);
    return CreatePlacedResource(pHeap, Offset, &desc, InitialState, OptimizedClearValue, riid, ppResource);
  }

  void STDMETHODCALLTYPE
  CreateSamplerFeedbackUnorderedAccessView(ID3D12Resource *targeted, ID3D12Resource *feedback, D3D12_CPU_DESCRIPTOR_HANDLE dst) {
    DXMT_STAT_SCOPE("device.CreateSamplerFeedbackUnorderedAccessView");
    WARN("CreateSamplerFeedbackUnorderedAccessView: sampler feedback not supported");
  }

  void STDMETHODCALLTYPE
  GetCopyableFootprints1(
      const D3D12_RESOURCE_DESC1 *pDesc, UINT FirstSubresource, UINT SubresourceCount, UINT64 BaseOffset,
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT *pLayouts, UINT *pNumRows, UINT64 *pRowSizeInBytes, UINT64 *pTotalBytes
  ) {
    DXMT_STAT_SCOPE("device.GetCopyableFootprints1");
    auto desc = PlainDesc(*pDesc);
    GetCopyableFootprints(&desc, FirstSubresource, SubresourceCount, BaseOffset, pLayouts, pNumRows, pRowSizeInBytes, pTotalBytes);
  }

  HRESULT STDMETHODCALLTYPE
  CreateHeap1(const D3D12_HEAP_DESC *pDesc, ID3D12ProtectedResourceSession *pSession, REFIID riid, void **ppHeap) {
    DXMT_STAT_SCOPE("device.CreateHeap1");
    if (pSession)
      return E_NOTIMPL; // protected sessions: none, as CreateCommittedResource1
    return CreateHeap(pDesc, riid, ppHeap);
  }

  HRESULT STDMETHODCALLTYPE
  CreateReservedResource1(
      const D3D12_RESOURCE_DESC *pDesc, D3D12_RESOURCE_STATES InitialState,
      const D3D12_CLEAR_VALUE *OptimizedClearValue, ID3D12ProtectedResourceSession *pSession, REFIID riid,
      void **ppResource
  ) {
    DXMT_STAT_SCOPE("device.CreateReservedResource1");
    return E_NOTIMPL;
  }

  D3D12_RESOURCE_ALLOCATION_INFO *STDMETHODCALLTYPE
  GetResourceAllocationInfo1(
      D3D12_RESOURCE_ALLOCATION_INFO *__ret, UINT VisibleMask, UINT ResourceDestCount,
      const D3D12_RESOURCE_DESC *pDescs, D3D12_RESOURCE_ALLOCATION_INFO1 *pAllocationInfos
  ) {
    DXMT_STAT_SCOPE("device.GetResourceAllocationInfo1");
    D3D12_RESOURCE_ALLOCATION_INFO1 resource_info;
    bool has_msaa_resource = false;

    __ret->SizeInBytes = 0;
    __ret->Alignment = 1;

    for (unsigned i = 0; i < ResourceDestCount; i++) {
      const D3D12_RESOURCE_DESC *desc = &pDescs[i];
      has_msaa_resource |= desc->SampleDesc.Count > 1;

      if (desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
        if (desc->Alignment && desc->Alignment != D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT) {
          DEBUG("GetResourceAllocationInfo: invalid alignment ", desc->Alignment, " for buffer resource.\n");
          goto invalid;
        }
        resource_info.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        auto size_and_align = GetMTLDevice().heapBufferSizeAndAlign(desc->Width, {});
        resource_info.SizeInBytes = size_and_align.size;
      } else {
        WMTTextureInfo texture_info;
        if (FAILED(PopulateWMTTextureInfo(GetMTLDevice(), texture_info, *desc))) {
          DEBUG("GetResourceAllocationInfo: invalid texture descriptor\n");
          goto invalid;
        }
        auto size_and_align = GetMTLDevice().heapTextureSizeAndAlign(texture_info);
        resource_info.SizeInBytes = size_and_align.size;
        resource_info.Alignment = size_and_align.align;
        auto requested_alignment = desc->Alignment              ? desc->Alignment
                                   : desc->SampleDesc.Count > 1 ? D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT
                                                                : D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        resource_info.Alignment = std::max(resource_info.Alignment, requested_alignment);
      }

      resource_info.SizeInBytes = align(resource_info.SizeInBytes, resource_info.Alignment);
      resource_info.Offset = align(__ret->SizeInBytes, resource_info.Alignment);

      if (pAllocationInfos)
        pAllocationInfos[i] = resource_info;

      __ret->SizeInBytes = resource_info.Offset + resource_info.SizeInBytes;
      __ret->Alignment = std::max(__ret->Alignment, resource_info.Alignment);
    }

    __ret->SizeInBytes = align(__ret->SizeInBytes, __ret->Alignment);
    return __ret;

  invalid:

    __ret->SizeInBytes = ~(uint64_t)0;
    __ret->Alignment = has_msaa_resource ? D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT
                                         : D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    return __ret;
  }

  WMT::ResidencySet
  GetGlobalResidencySet() {
    return residency_set_;
  };

  HRESULT
  RegisterResidency(WMT::Allocation allocation) {
    std::unique_lock<dxmt::mutex> lock(residency_lock_);
    residency_set_.addAllocations(&allocation, 1);
    residency_set_.commit();
    return S_OK;
  }

  HRESULT
  UnregisterResidency(WMT::Allocation allocation) {
    std::unique_lock<dxmt::mutex> lock(residency_lock_);
    residency_set_.removeAllocations(&allocation, 1);
    residency_set_.commit();
    return S_OK;
  }

  std::pair<Texture *, TextureViewKey>
  NullTexture(WMTTextureType type) {
    std::lock_guard<dxmt::mutex> lock(null_lock_);
    auto &texture = null_textures_[type];
    if (!texture) {
      bool multisample = type == WMTTextureType2DMultisample || type == WMTTextureType2DMultisampleArray;
      WMTTextureInfo info = {};
      info.pixel_format = WMTPixelFormatRGBA32Float; // all four channels read 0
      info.type = type;
      info.width = info.height = info.depth = 1;
      info.array_length = 1;
      info.mipmap_level_count = 1;
      info.sample_count = multisample ? 2 : 1; // the sizes D3DMetal's null views report
      info.usage = WMTTextureUsage(WMTTextureUsageShaderRead | WMTTextureUsagePixelFormatView);
      // Shared, so it can be zeroed here; multisample textures can't be, and rely on fresh memory being zero.
      info.options = WMTResourceOptions(
          (multisample ? WMTResourceStorageModePrivate : WMTResourceStorageModeShared) | WMTResourceHazardTrackingModeUntracked
      );
      texture = new Texture(info, GetMTLDevice());
      texture->rename(texture->allocate({}));
      RegisterResidency(texture->current()->texture());
      if (!multisample) {
        const float zero[4] = {};
        unsigned slices = type == WMTTextureTypeCube || type == WMTTextureTypeCubeArray ? 6 : 1;
        for (unsigned slice = 0; slice < slices; slice++)
          texture->current()->texture().replaceRegion({0, 0, 0}, {1, 1, 1}, 0, slice, zero, sizeof(zero), sizeof(zero));
      }
    }
    return {texture.ptr(), texture->fullView};
  }

  uint64_t
  TimestampFrequency() {
    // GPU ticks against QueryPerformanceCounter (a known rate) 10 ms apart, once.
    std::call_once(timestamp_once_, [&] {
      uint64_t cpu0, gpu0, cpu1, gpu1;
      LARGE_INTEGER q0, q1, qf;
      QueryPerformanceFrequency(&qf);
      QueryPerformanceCounter(&q0);
      GetMTLDevice().sampleTimestamps(cpu0, gpu0);
      Sleep(10);
      QueryPerformanceCounter(&q1);
      GetMTLDevice().sampleTimestamps(cpu1, gpu1);
      timestamp_frequency_ =
          gpu1 > gpu0 && q1.QuadPart > q0.QuadPart
              ? (uint64_t)((double)(gpu1 - gpu0) * qf.QuadPart / (q1.QuadPart - q0.QuadPart) + 0.5)
              : 1;
    });
    return timestamp_frequency_;
  }

  std::pair<Buffer *, BufferViewKey>
  NullTexelBuffer() {
    std::lock_guard<dxmt::mutex> lock(null_lock_);
    if (!null_buffer_) {
      null_buffer_ = new Buffer(16, GetMTLDevice());
      null_buffer_->rename(null_buffer_->allocate({}));
      memset(null_buffer_->current()->mappedMemory(0), 0, 16);
      RegisterResidencyAndVA(null_buffer_->current());
      null_buffer_view_ = null_buffer_->createView(BufferViewDescriptor{WMTPixelFormatRGBA32Float});
    }
    return {null_buffer_.ptr(), null_buffer_view_};
  }

  HRESULT
  RegisterResidencyAndVA(BufferAllocation *allocation) {
    std::unique_lock<dxmt::mutex> lock(residency_lock_);
    interval_map_.emplace(allocation->gpuAddress(), allocation);
    if (allocation->flags().test(BufferAllocationFlag::AllocatedOnHeap))
      return S_OK;
    auto buffer = allocation->buffer();
    residency_set_.addAllocations(&buffer, 1);
    residency_set_.commit();
    return S_OK;
  }

  HRESULT
  UnregisterResidencyAndVA(BufferAllocation *allocation) {
    std::unique_lock<dxmt::mutex> lock(residency_lock_);
    interval_map_.erase(allocation->gpuAddress());
    if (allocation->flags().test(BufferAllocationFlag::AllocatedOnHeap))
      return S_OK;
    auto buffer = allocation->buffer();
    residency_set_.removeAllocations(&buffer, 1);
    residency_set_.commit();
    return S_OK;
  }

  BufferAllocation *
  LookupBufferByVA(D3D12_GPU_VIRTUAL_ADDRESS VA, uint64_t *pOffset) {
    std::unique_lock<dxmt::mutex> lock(residency_lock_);
    auto iter = interval_map_.upper_bound(VA);
    if (iter == interval_map_.begin()) {
      *pOffset = 0;
      return {};
    }
    --iter;
    *pOffset = VA - iter->first;
    return iter->second;
  }

  InternalCommandLibrary &
  GetLib() {
    return command_library;
  }

  virtual FormatCapability
  GetMTLPixelFormatCapability(WMTPixelFormat Format) final {
    Format = ORIGINAL_FORMAT(Format);
    if (!format_inspector_.textureCapabilities.contains(Format))
      return FormatCapability(0);
    return format_inspector_.textureCapabilities.at(Format);
  };
};

HRESULT
CreateD3D12Device(IMTLDXGIAdapter *adapter, const IID &riid, void **ppDevice) {
  auto device = Com(new MTLD3D12DeviceImpl(adapter));
  HRESULT hr = device->Initialize();
  if (FAILED(hr))
    return hr;
  return device->QueryInterface(riid, ppDevice);
};

} // namespace dxmt