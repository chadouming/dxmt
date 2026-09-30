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

#include "com/com_pointer.hpp"
#include "d3d12_pageable.hpp"
#include <algorithm>
#include <cstring>

namespace dxmt {

class MTLD3D12QueryHeapImpl : public MTLD3D12Pageable<MTLD3D12QueryHeap> {
public:
  MTLD3D12QueryHeapImpl(MTLD3D12Device *pDevice) : MTLD3D12Pageable<MTLD3D12QueryHeap>(pDevice) {}

  HRESULT
  Initialize(const D3D12_QUERY_HEAP_DESC *pDesc) {
    switch (pDesc->Type) {
    case D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS:
      stride = sizeof(D3D12_QUERY_DATA_PIPELINE_STATISTICS);
      break;
    case D3D12_QUERY_HEAP_TYPE_SO_STATISTICS:
      stride = sizeof(D3D12_QUERY_DATA_SO_STATISTICS);
      break;
    default:
      stride = sizeof(UINT64);
      break;
    }
    // Zeroed: only occlusion queries write results; the other types resolve to zeros.
    WMTBufferInfo info;
    info.length = std::max(1u, pDesc->Count) * (uint64_t)stride;
    info.options = WMTResourceStorageModeShared; // tracked: the passes counting into it, then resolves, in order
    info.memory.set(0);
    results = device_->GetMTLDevice().newBuffer(info);
    if (!results || !info.memory.get_accessible_or_null())
      return E_OUTOFMEMORY;
    memset(info.memory.get_accessible_or_null(), 0, info.length);
    if (pDesc->Type == D3D12_QUERY_HEAP_TYPE_TIMESTAMP) {
      auto metal = device_->GetMTLDevice();
      for (uint32_t first = 0; first < pDesc->Count; first += kTimestampsPerBuffer) {
        // Shared: the queue resolves it on the CPU once the samples are written (see TimestampResolve).
        auto samples = metal.newCounterSampleBuffer(std::min(kTimestampsPerBuffer, pDesc->Count - first), true);
        if (!samples) { // no stage-boundary sampling: timestamps resolve to zeros, as before
          counters.clear();
          break;
        }
        counters.push_back(std::move(samples));
      }
      aliases.assign(pDesc->Count, ~0u);
    }
    return S_OK;
  }

  HRESULT
  STDMETHODCALLTYPE
  QueryInterface(REFIID riid, void **ppvObject) {
    if (ppvObject == nullptr)
      return E_POINTER;

    *ppvObject = nullptr;

    if (riid == __uuidof(IUnknown) || riid == __uuidof(ID3D12Object) || riid == __uuidof(ID3D12DeviceChild) ||
        riid == __uuidof(ID3D12Pageable) || riid == __uuidof(ID3D12QueryHeap)) {
      *ppvObject = ref(this);
      return S_OK;
    }

    if (logQueryInterfaceError(__uuidof(ID3D12QueryHeap), riid)) {
      WARN("D3D12QueryHeap: Unknown interface query ", str::format(riid));
    }

    return E_NOINTERFACE;
  }
};

HRESULT
CreateQueryHeap(MTLD3D12Device *pDevice, const D3D12_QUERY_HEAP_DESC *pDesc, REFIID riid, void **ppQueryHeap) {
  auto heap = Com(new MTLD3D12QueryHeapImpl(pDevice));
  HRESULT hr = heap->Initialize(pDesc);
  if (FAILED(hr))
    return hr;
  if (!ppQueryHeap)
    return S_FALSE;
  return heap->QueryInterface(riid, ppQueryHeap);
}

} // namespace dxmt