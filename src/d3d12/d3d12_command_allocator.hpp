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

#pragma once

#include "d3d12_pageable.hpp"
#include "dxmt_command_clear.hpp"
#include "dxmt_ring_bump_allocator.hpp"
#include <algorithm>

namespace dxmt {

constexpr auto kCPUHeapSize = 0x400000u;
constexpr auto kGPUHeapSize = 0x400000u;

inline std::size_t
align_forward_adjustment(const void *const ptr, const std::size_t &alignment) noexcept {
  const auto iptr = reinterpret_cast<std::uintptr_t>(ptr);
  const auto aligned = (iptr - 1u + alignment) & -alignment;
  return aligned - iptr;
}

inline void *
ptr_add(const void *const p, const std::uintptr_t &amount) noexcept {
  return reinterpret_cast<void *>(reinterpret_cast<std::uintptr_t>(p) + amount);
}

struct IndirectComputeCommandData {
  uint64_t cmd_buf;
  uint64_t max_count;
  uint64_t max_count_buffer;
  uint64_t argument_buffer;
  uint64_t static_samplers;
  uint64_t rootsig_qwords;
  uint32_t rootsig_qwords_stride;
  uint32_t tgsize_x;
  uint32_t tgsize_y;
  uint32_t tgsize_z;
};

struct IndirectRenderCommandData {
  uint64_t cmd_buf;
  uint64_t max_count;
  uint64_t max_count_buffer;
  uint64_t argument_buffer;
  uint64_t static_samplers;
  uint64_t rootsig_qwords;
  uint32_t rootsig_qwords_stride;
  uint32_t primitive_type;
  uint64_t vertex_buffer;
  uint64_t index_buffer;
  DXGI_FORMAT index_buffer_format;
  uint32_t vertex_argbuf_stride;
};

class MTLD3D12CommandAllocatorImpl : public MTLD3D12Pageable<MTLD3D12CommandAllocator> {
  friend class MTLD3D12GraphicsCommandListImpl;
  friend struct SimpleCommandContext<MTLD3D12CommandAllocatorImpl>;

  D3D12_COMMAND_LIST_TYPE type_;

  std::vector<void *> spilled_cpu_heap_;
  void *cpu_heap_ = nullptr;
  size_t cpu_heap_offset_;

  WMT::Reference<WMT::Buffer> gpu_heap_buffer_;
  void *gpu_heap_ = nullptr;
  size_t gpu_heap_offset_;
  uint64_t gpu_heap_buffer_address_;

  EncoderData *encoder_last;
  EncoderData *encoder_current;
  size_t encoder_count_;
  uint32_t barriers_ = 0; // this list's ResourceBarrier calls (and GPU query resolves, which order as one)

  // MacNeutron: encoder ordering (GPU overlap spec §3.1-3.2), decided as each encoder closes. joins_[n]: how many of
  // the list's first n barrier calls order all earlier work before all later work (M1: every one). group_: the
  // list's encoders since its last join.
  std::vector<uint32_t> joins_ = {0};
  std::vector<EncoderData *> group_;
  std::vector<uint32_t> deps_; // scratch

  small_vector<EncoderData, 64> encoder_lists_;

  // MacNeutron: ExecuteIndirect's command buffers, kept by kind and capacity and reused once Reset says the GPU is
  // done with them (creating and releasing one per call cost ~25 us each under Rosetta).
  struct PooledICB {
    uint64_t key;
    WMT::Reference<WMT::IndirectCommandBuffer> icb;
    uint64_t resource_id;
  };
  std::vector<PooledICB> icb_;
  std::unordered_map<uint64_t, std::vector<PooledICB>> icb_free_; // by key; no allocation once warm
  PooledICB &AcquireICB(WMTIndirectCommandBufferInfo &info, size_t MaxCount, WMTResourceOptions storage);

  ClearUAV<MTLD3D12CommandAllocatorImpl> clear_uav_;
  ClearRTV<MTLD3D12CommandAllocatorImpl> clear_rtv_;

  RingBumpState<GpuPrivateBufferBlockAllocator> copy_temp_allocator_;
  uint64_t copy_temp_version_;

public:
  MTLD3D12CommandAllocatorImpl(MTLD3D12Device *pDevice, D3D12_COMMAND_LIST_TYPE Type);

  ~MTLD3D12CommandAllocatorImpl() {
    free(cpu_heap_);
    cpu_heap_ = nullptr;
    gpu_heap_buffer_ = {};
    free(gpu_heap_);
    gpu_heap_ = nullptr;
  }

  HRESULT
  Initialize();

  HRESULT
  STDMETHODCALLTYPE
  QueryInterface(REFIID riid, void **ppvObject);

  HRESULT STDMETHODCALLTYPE Reset();

  HRESULT STDMETHODCALLTYPE CreateCommandList(
      UINT NodeMask, D3D12_COMMAND_LIST_TYPE Type, ID3D12PipelineState *pInitialPipelineState, REFIID riid,
      void **ppCommandList
  );

  void
  InvalidateCurrentPass() {
    if (!encoder_current)
      return;
    Decide(encoder_current);
    encoder_last->next = encoder_current;
    encoder_last = encoder_current;

    encoder_current = nullptr;
    encoder_count_++;
  }

  HRESULT
  StartRecord(EncoderData **pStartEncoder) {
    if (encoder_last)
      return E_INVALIDARG;
    barriers_ = 0;
    joins_.assign(1, 0);
    group_.clear();
    *pStartEncoder = encoder_last = &encoder_lists_.emplace_back(EncoderType::Null, nullptr);
    return S_OK;
  }

  HRESULT
  EndRecord(size_t *pEncoderCount) {
    if (!encoder_last)
      return E_FAIL;
    if (encoder_current)
      InvalidateCurrentPass();
    encoder_last = nullptr;
    *pEncoderCount = encoder_count_;
    encoder_count_ = 0;
    return S_OK;
  }

  void *
  AllocateCPUHeap(size_t Length, size_t Alignment) {
    assert(Length < kCPUHeapSize);
    for (;;) {
      std::size_t adjustment = align_forward_adjustment((void *)cpu_heap_offset_, Alignment);
      auto aligned = cpu_heap_offset_ + adjustment;
      cpu_heap_offset_ = aligned + Length;
      if (unlikely(cpu_heap_offset_ >= kCPUHeapSize)) {
        spilled_cpu_heap_.push_back(cpu_heap_);
        cpu_heap_ = malloc(kCPUHeapSize);
        cpu_heap_offset_ = 0;
        continue;
      }
      return ptr_add(cpu_heap_, aligned);
    }
  }

  template <typename T>
  T *
  AllocatePass() {
    auto p = (new (AllocateCPUHeap(sizeof(T), alignof(T))) T());
    p->barriers = p->barriers_last = barriers_;
    encoder_current = p;
    return p;
  };

  // A barrier call: `join` when it orders all earlier work before all later work.
  void
  Barrier(bool join) {
    barriers_++;
    joins_.push_back(joins_.back() + join);
  }

  // MacNeutron: the current encoder writes `resource` (GPU overlap spec §3.2): a DXMT Texture or Buffer object, or a
  // query heap's Metal results buffer. Null, or more than an encoder lists: it may write anything.
  void
  Writes(const void *resource) {
    auto e = encoder_current;
    if (!resource || e->write_count == std::size(e->writes)) {
      e->writes_unknown = true;
      return;
    }
    for (unsigned i = 0; i < e->write_count; i++)
      if (e->writes[i] == resource)
        return;
    e->writes[e->write_count++] = resource;
  }

  static bool
  Overlap(const EncoderData *a, const EncoderData *b) {
    for (unsigned i = 0; i < a->write_count; i++)
      for (unsigned j = 0; j < b->write_count; j++)
        if (a->writes[i] == b->writes[j])
          return true;
    return false;
  }

  // What the closing encoder `e` waits on (GPU overlap spec §3.1). A join (all earlier work) when it is the list's
  // first, may write anything, or a joining barrier came after the previous encoder began and before e's last
  // command. Otherwise, the encoders since the list's last join that write what it writes or may write anything.
  // ponytail: every earlier writer, not only the newest, so waits grow with same-target passes in one group; keep
  // the newest writer per resource if dependency waits show in DXMT_STATS.
  void
  Decide(EncoderData *e) {
    auto prev = encoder_last; // the list's previous encoder, or its Null head
    e->position = encoder_count_;
    e->join = prev->type == EncoderType::Null || e->writes_unknown || joins_[e->barriers_last] > joins_[prev->barriers];
    if (e->join) {
      group_.clear();
    } else {
      deps_.clear();
      for (auto *g : group_)
        if (g->writes_unknown || Overlap(g, e))
          deps_.push_back(g->position);
      if (!deps_.empty()) {
        auto deps = AllocateCommandData<uint32_t>(deps_.size());
        std::copy(deps_.begin(), deps_.end(), deps);
        e->deps = deps;
        e->dep_count = deps_.size();
      }
    }
    group_.push_back(e);
  }

  template <typename T>
  T *
  AllocateCommandData(size_t Count) {
    return (T *)AllocateCPUHeap(sizeof(T) * Count, alignof(T));
  };

  // MacNeutron: compute work the queue runs just before the current render pass (ExecuteIndirect's resolvers).
  template <typename cmd_struct>
  cmd_struct &
  EncodeRenderPreCommand() {
    assert(encoder_current->type == EncoderType::Render);
    auto encoder = static_cast<RenderEncoderData *>(encoder_current);
    if (!encoder->pre_tail) {
      encoder->pre_head.type = WMTComputeCommandNop;
      encoder->pre_head.next.set(nullptr);
      encoder->pre_tail = (wmtcmd_base *)&encoder->pre_head;
    }
    auto storage = (cmd_struct *)AllocateCPUHeap(sizeof(cmd_struct), 16);
    encoder->pre_tail->next.set(storage);
    encoder->pre_tail = (wmtcmd_base *)storage;
    storage->next.set(nullptr);
    encoder->barriers_last = barriers_; // a barrier recorded before this command binds the whole encoder
    return *storage;
  }

  template <typename cmd_struct>
  cmd_struct &
  EncodeRenderCommand() {
    assert(encoder_current->type == EncoderType::Render);
    auto encoder = static_cast<RenderEncoderData *>(encoder_current);
    auto storage = (cmd_struct *)AllocateCPUHeap(sizeof(cmd_struct), 16);
    encoder->cmd_tail->next.set(storage);
    encoder->cmd_tail = (wmtcmd_base *)storage;
    storage->next.set(nullptr);
    encoder->barriers_last = barriers_; // a barrier recorded before this command binds the whole encoder
    return *storage;
  }

  template <typename cmd_struct>
  cmd_struct &
  EncodeBlitCommand() {
    assert(encoder_current->type == EncoderType::Blit);
    auto encoder = static_cast<BlitEncoderData *>(encoder_current);
    auto storage = (cmd_struct *)AllocateCPUHeap(sizeof(cmd_struct), 16);
    encoder->cmd_tail->next.set(storage);
    encoder->cmd_tail = (wmtcmd_base *)storage;
    storage->next.set(nullptr);
    encoder->barriers_last = barriers_; // a barrier recorded before this command binds the whole encoder
    return *storage;
  }

  template <typename cmd_struct>
  cmd_struct &
  EncodeComputeCommand() {
    assert(encoder_current->type == EncoderType::Compute);
    auto encoder = static_cast<ComputeEncoderData *>(encoder_current);
    auto storage = (cmd_struct *)AllocateCPUHeap(sizeof(cmd_struct), 16);
    encoder->cmd_tail->next.set(storage);
    encoder->cmd_tail = (wmtcmd_base *)storage;
    storage->next.set(nullptr);
    encoder->barriers_last = barriers_; // a barrier recorded before this command binds the whole encoder
    return *storage;
  }

  std::tuple<void *, size_t>
  AllocateGPUHeap(size_t Length, size_t Alignment) {
    if (!Length)
      return {nullptr, 0};
    std::size_t adjustment = align_forward_adjustment((void *)gpu_heap_offset_, Alignment);
    auto aligned = gpu_heap_offset_ + adjustment;
    gpu_heap_offset_ = aligned + Length;
    assert(gpu_heap_offset_ < kGPUHeapSize);
    return {ptr_add(gpu_heap_, aligned), aligned};
  }

  std::tuple<WMT::Buffer, uint64_t>
  AllocateTempBuffer(size_t Size, size_t Alignment) {
    assert(copy_temp_version_);
    auto [block, offset] = copy_temp_allocator_.allocate(copy_temp_version_, copy_temp_version_ - 1, Size, Alignment);
    return {block.buffer, offset};
  }

  IndirectComputeCommandData *EncodeIndirectComputeCommand(MTLD3D12CommandSignature *pCmdSig, MTLD3D12ComputePipelineState *pPSO, size_t MaxCount);

  IndirectRenderCommandData *EncodeIndirectRenderCommand(MTLD3D12CommandSignature *pCmdSig, MTLD3D12GraphicsPipelineState *pPSO, size_t MaxCount);
};

} // namespace dxmt