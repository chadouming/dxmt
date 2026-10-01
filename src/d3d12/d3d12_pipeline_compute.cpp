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

#include "Metal.hpp"
#include "d3d12_stats.hpp"
#include "com/com_pointer.hpp"
#include "d3d12_device.hpp"
#include "d3d12_pageable.hpp"
#include "d3d12_pipeline.hpp"
#include "d3d12_shader_cache.hpp"
#include "log/log.hpp"
#include "d3d12_dxil_dump.hpp"
#include "../d3d10/d3d10_blob.hpp"

namespace dxmt {

class MTLD3D12ComputePipelineStateImpl : public MTLD3D12Pageable<MTLD3D12ComputePipelineState> {

  MTL_SHADER_REFLECTION ref_cs;
  Sha1Digest digest_cs_ = {}; // for the recording

public:
  MTLD3D12ComputePipelineStateImpl(MTLD3D12Device *pDevice) : MTLD3D12Pageable<MTLD3D12ComputePipelineState>(pDevice) {
    IsComputePipelineState = 1;
  }

  HRESULT
  Initialize(const D3D12_COMPUTE_PIPELINE_STATE_DESC *pDesc) {
    DumpDXIL(pDesc->CS);
    if (DXILCaptureMode())
      LogPipeline("cs" + CapturedShader("cs", pDesc->CS) + CapturedRootSignature(pDesc->pRootSignature));

    CachedShader shader_cs;

    SM50_SHADER_ROOT_SIGNATURE_DATA rootsig;
    rootsig.type = SM50_SHADER_ROOT_SIGNATURE;
    if (pDesc->pRootSignature) {
      rootsig.bytecode_length = static_cast<MTLD3D12RootSignature *>(pDesc->pRootSignature)->GetBlob(&rootsig.bytecode);
    } else {
      rootsig.bytecode = pDesc->CS.pShaderBytecode;
      rootsig.bytecode_length = pDesc->CS.BytecodeLength;
    }
    rootsig.next = nullptr;

    SM50_SHADER_COMMON_DATA common;
    common.flags = {};
    common.type = SM50_SHADER_COMMON;
    common.metal_version = SM50_SHADER_METAL_310;
    common.next = &rootsig;

    HRESULT hr;
    if (FAILED(hr = shader_cs.Initialize(pDesc->CS)) || FAILED(hr = shader_cs.Reflection(&ref_cs)))
      return hr;

    threadgroup_size = {ref_cs.ThreadgroupSize[0], ref_cs.ThreadgroupSize[1], ref_cs.ThreadgroupSize[2]};

    auto metal = device_->GetMTLDevice();

    WMT::Reference<WMT::Error> err;

    WMT::Reference<WMT::Function> cs_func;
    if (FAILED(hr = CompileFunction(metal, FunctionKind::Shader, shader_cs, nullptr,
                                    (SM50_SHADER_COMPILATION_ARGUMENT_DATA *)&common, "cs_main", "cs", cs_func)))
      return hr;
    digest_cs_ = shader_cs.digest();

    // PSO
    {
      WMTComputePipelineInfo info;
      WMT::InitializeComputePipelineInfo(info);
      info.compute_function = cs_func;
      info.support_indirect_command_buffers = true;

      pso = metal.newComputePipelineState(info, err);
      if (!pso) {
        ERR("Failed to create compute PSO: ", err.description().getUTF8String());
        return E_FAIL;
      }
    }

    return S_OK;
  }

  // Shader pre-caching: appends this pipeline to the recording (spec §3.5), once its creation succeeded.
  void
  Record(const D3D12_COMPUTE_PIPELINE_STATE_DESC &desc) {
    if (record::RecordingOn())
      record::RecordCompute(desc, RootBlob(desc.pRootSignature),
                            {digest_cs_, desc.CS.pShaderBytecode, desc.CS.BytecodeLength});
  }

  HRESULT
  STDMETHODCALLTYPE
  QueryInterface(REFIID riid, void **ppvObject) {
    DXMT_STAT_SCOPE("computepso.QueryInterface");
    if (ppvObject == nullptr)
      return E_POINTER;

    *ppvObject = nullptr;

    if (riid == __uuidof(IUnknown) || riid == __uuidof(ID3D12Object) || riid == __uuidof(ID3D12DeviceChild) ||
        riid == __uuidof(ID3D12Pageable) || riid == __uuidof(ID3D12PipelineState)) {
      *ppvObject = ref(this);
      return S_OK;
    }

    if (logQueryInterfaceError(__uuidof(ID3D12PipelineState), riid)) {
      WARN("D3D12ComputePipelineState: Unknown interface query ", str::format(riid));
    }

    return E_NOINTERFACE;
  }

  virtual HRESULT STDMETHODCALLTYPE
  GetCachedBlob(ID3DBlob **blob) {
    DXMT_STAT_SCOPE("computepso.GetCachedBlob");
    // An empty blob, as D3DMetal: a CachedPSO is accepted and ignored at creation.
    if (!blob)
      return E_POINTER;
    return CreateBlobFromMalloc(0, blob);
  }
};

HRESULT
CreateComputePipelineState(
    MTLD3D12Device *pDevice, const D3D12_COMPUTE_PIPELINE_STATE_DESC *pDesc, REFIID riid, void **ppPipelineState
) {
  auto pso = Com(new MTLD3D12ComputePipelineStateImpl(pDevice));
  HRESULT hr = pso->Initialize(pDesc);
  if (FAILED(hr))
    return hr;
  pso->desc_hash = HashComputeDesc(*pDesc);
  pso->Record(*pDesc);
  return pso->QueryInterface(riid, ppPipelineState);
};

} // namespace dxmt