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

// Pipeline libraries (MacNeutron): D3DMetal's semantics, see the D3D12 stubs spec §3.1.7.
#include "d3d12_device.hpp"
#include "d3d12_stats.hpp"
#include "d3d12_device_child.hpp"
#include "com/com_pointer.hpp"
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <string_view>

namespace dxmt {

namespace {
// std::hash over bytes: fast, and stable within one DXMT build (a library serialized by another build no longer
// matches, and the app recreates its pipelines).
struct Hasher {
  uint64_t h = 0;
  void bytes(const void *p, size_t n) {
    h = h * 1099511628211ull ^ std::hash<std::string_view>{}(std::string_view((const char *)p, n));
  }
  template <typename T> void pod(const T &v) { bytes(&v, sizeof v); }
  void str(const char *s) { bytes(s ? s : "", s ? strlen(s) : 0); }
  void shader(const D3D12_SHADER_BYTECODE &b) { bytes(b.pShaderBytecode, b.pShaderBytecode ? b.BytecodeLength : 0); }
  void root(ID3D12RootSignature *rs) {
    const void *blob = nullptr;
    size_t size = rs ? static_cast<MTLD3D12RootSignature *>(rs)->GetBlob(&blob) : 0;
    bytes(blob, size);
  }
};
} // namespace

uint64_t
HashGraphicsDesc(const D3D12_GRAPHICS_PIPELINE_STATE_DESC &d) {
  Hasher x;
  x.root(d.pRootSignature);
  for (auto *s : {&d.VS, &d.PS, &d.DS, &d.HS, &d.GS})
    x.shader(*s);
  for (UINT i = 0; i < d.StreamOutput.NumEntries; i++) {
    auto &e = d.StreamOutput.pSODeclaration[i];
    x.str(e.SemanticName);
    x.pod(e.Stream); x.pod(e.SemanticIndex); x.pod(e.StartComponent); x.pod(e.ComponentCount); x.pod(e.OutputSlot);
  }
  x.bytes(d.StreamOutput.pBufferStrides, d.StreamOutput.NumStrides * sizeof(UINT));
  x.pod(d.StreamOutput.RasterizedStream);
  x.pod(d.BlendState); x.pod(d.SampleMask); x.pod(d.RasterizerState); x.pod(d.DepthStencilState);
  for (UINT i = 0; i < d.InputLayout.NumElements; i++) {
    auto &e = d.InputLayout.pInputElementDescs[i];
    x.str(e.SemanticName);
    x.pod(e.SemanticIndex); x.pod(e.Format); x.pod(e.InputSlot); x.pod(e.AlignedByteOffset);
    x.pod(e.InputSlotClass); x.pod(e.InstanceDataStepRate);
  }
  x.pod(d.IBStripCutValue); x.pod(d.PrimitiveTopologyType); x.pod(d.NumRenderTargets); x.pod(d.RTVFormats);
  x.pod(d.DSVFormat); x.pod(d.SampleDesc); x.pod(d.NodeMask); x.pod(d.Flags);
  return x.h;
}

uint64_t
HashComputeDesc(const D3D12_COMPUTE_PIPELINE_STATE_DESC &d) {
  Hasher x;
  x.root(d.pRootSignature);
  x.shader(d.CS);
  x.pod(d.NodeMask); x.pod(d.Flags);
  return x.h ^ 1; // never equal to a graphics description's
}

namespace {
constexpr char kMagic[8] = {'D', 'X', 'M', 'T', 'P', 'L', 'B', '1'};

class MTLD3D12PipelineLibraryImpl : public MTLD3D12DeviceChild<ID3D12PipelineLibrary1> {
  struct Entry {
    uint64_t hash;
    Com<ID3D12PipelineState> pso; // null for a name read from a serialized library until its first load
  };
  std::mutex mutex_;
  std::map<std::wstring, Entry> entries_;

  // A stored or serialized name with this description's hash: the pipeline, created on first load.
  HRESULT
  Load(LPCWSTR name, uint64_t hash, const std::function<HRESULT(ID3D12PipelineState **)> &create, REFIID riid, void **pp) {
    InitReturnPtr(pp);
    if (!name)
      return E_INVALIDARG;
    std::lock_guard lock(mutex_);
    auto it = entries_.find(name);
    if (it == entries_.end() || it->second.hash != hash)
      return E_INVALIDARG;
    if (!it->second.pso) {
      ID3D12PipelineState *pso = nullptr;
      HRESULT hr = create(&pso);
      if (FAILED(hr))
        return hr;
      it->second.pso = pso;
      pso->Release(); // the Com holds it
    }
    return it->second.pso->QueryInterface(riid, pp);
  }

public:
  MTLD3D12PipelineLibraryImpl(MTLD3D12Device *pDevice) : MTLD3D12DeviceChild<ID3D12PipelineLibrary1>(pDevice) {}

  HRESULT
  Initialize(const void *blob, SIZE_T size) {
    if (!blob || !size)
      return S_OK;
    auto p = static_cast<const char *>(blob), end = p + size;
    uint32_t version, count;
    if (size < 16 || memcmp(p, kMagic, 8))
      return D3D12_ERROR_DRIVER_VERSION_MISMATCH;
    memcpy(&version, p + 8, 4);
    memcpy(&count, p + 12, 4);
    if (version != 1)
      return D3D12_ERROR_DRIVER_VERSION_MISMATCH;
    p += 16;
    for (uint32_t i = 0; i < count; i++) {
      uint32_t chars;
      if (end - p < 4)
        return E_INVALIDARG;
      memcpy(&chars, p, 4);
      p += 4;
      if ((size_t)(end - p) < chars * sizeof(WCHAR) + 8)
        return E_INVALIDARG;
      std::wstring name((const wchar_t *)p, chars);
      p += chars * sizeof(WCHAR);
      uint64_t hash;
      memcpy(&hash, p, 8);
      p += 8;
      entries_[name] = {hash, nullptr};
    }
    return S_OK;
  }

  HRESULT STDMETHODCALLTYPE
  QueryInterface(REFIID riid, void **ppvObject) {
    DXMT_STAT_SCOPE("library.QueryInterface");
    if (!ppvObject)
      return E_POINTER;
    *ppvObject = nullptr;
    if (riid == __uuidof(IUnknown) || riid == __uuidof(ID3D12Object) || riid == __uuidof(ID3D12DeviceChild) ||
        riid == __uuidof(ID3D12PipelineLibrary) || riid == __uuidof(ID3D12PipelineLibrary1)) {
      *ppvObject = ref(this);
      return S_OK;
    }
    return E_NOINTERFACE;
  }

  HRESULT STDMETHODCALLTYPE
  StorePipeline(LPCWSTR pName, ID3D12PipelineState *pPipeline) {
    DXMT_STAT_SCOPE("library.StorePipeline");
    if (!pName || !pPipeline)
      return E_INVALIDARG;
    std::lock_guard lock(mutex_);
    if (entries_.count(pName))
      return E_INVALIDARG;
    entries_[pName] = {static_cast<MTLD3D12PipelineState *>(pPipeline)->desc_hash, pPipeline};
    return S_OK;
  }

  HRESULT STDMETHODCALLTYPE
  LoadGraphicsPipeline(LPCWSTR pName, const D3D12_GRAPHICS_PIPELINE_STATE_DESC *pDesc, REFIID riid, void **pp) {
    DXMT_STAT_SCOPE("library.LoadGraphicsPipeline");
    if (!pDesc)
      return E_INVALIDARG;
    return Load(pName, HashGraphicsDesc(*pDesc), [&](ID3D12PipelineState **pso) {
      return device_->CreateGraphicsPipelineState(pDesc, __uuidof(ID3D12PipelineState), (void **)pso);
    }, riid, pp);
  }

  HRESULT STDMETHODCALLTYPE
  LoadComputePipeline(LPCWSTR pName, const D3D12_COMPUTE_PIPELINE_STATE_DESC *pDesc, REFIID riid, void **pp) {
    DXMT_STAT_SCOPE("library.LoadComputePipeline");
    if (!pDesc)
      return E_INVALIDARG;
    return Load(pName, HashComputeDesc(*pDesc), [&](ID3D12PipelineState **pso) {
      return device_->CreateComputePipelineState(pDesc, __uuidof(ID3D12PipelineState), (void **)pso);
    }, riid, pp);
  }

  HRESULT STDMETHODCALLTYPE
  LoadPipeline(LPCWSTR pName, const D3D12_PIPELINE_STATE_STREAM_DESC *pDesc, REFIID riid, void **pp) {
    DXMT_STAT_SCOPE("library.LoadPipeline");
    D3D12_GRAPHICS_PIPELINE_STATE_DESC graphics{};
    D3D12_COMPUTE_PIPELINE_STATE_DESC compute{};
    bool is_compute = false;
    HRESULT hr = pDesc ? ParsePipelineStream(pDesc, graphics, compute, is_compute) : E_INVALIDARG;
    if (FAILED(hr))
      return hr;
    return is_compute ? LoadComputePipeline(pName, &compute, riid, pp) : LoadGraphicsPipeline(pName, &graphics, riid, pp);
  }

  SIZE_T STDMETHODCALLTYPE
  GetSerializedSize() {
    DXMT_STAT_SCOPE("library.GetSerializedSize");
    std::lock_guard lock(mutex_);
    return SerializedSize();
  }

  SIZE_T
  SerializedSize() { // under mutex_
    SIZE_T size = 16;
    for (auto &[name, entry] : entries_)
      size += 4 + name.size() * sizeof(WCHAR) + 8;
    return size;
  }


  HRESULT STDMETHODCALLTYPE
  Serialize(void *pData, SIZE_T DataSizeInBytes) {
    DXMT_STAT_SCOPE("library.Serialize");
    std::lock_guard lock(mutex_); // one lock for the size and the write: a store in between would overflow pData
    if (!pData || DataSizeInBytes < SerializedSize())
      return E_INVALIDARG;
    auto p = static_cast<char *>(pData);
    uint32_t version = 1, count = entries_.size();
    memcpy(p, kMagic, 8);
    memcpy(p + 8, &version, 4);
    memcpy(p + 12, &count, 4);
    p += 16;
    for (auto &[name, entry] : entries_) {
      uint32_t chars = name.size();
      memcpy(p, &chars, 4);
      memcpy(p + 4, name.data(), chars * sizeof(WCHAR));
      p += 4 + chars * sizeof(WCHAR);
      memcpy(p, &entry.hash, 8);
      p += 8;
    }
    return S_OK;
  }

};
} // namespace

HRESULT
CreatePipelineLibrary(MTLD3D12Device *pDevice, const void *blob, SIZE_T size, REFIID riid, void **ppLibrary) {
  InitReturnPtr(ppLibrary);
  auto library = Com(new MTLD3D12PipelineLibraryImpl(pDevice));
  HRESULT hr = library->Initialize(blob, size);
  if (FAILED(hr))
    return hr;
  return ppLibrary ? library->QueryInterface(riid, ppLibrary) : S_FALSE;
}

} // namespace dxmt
