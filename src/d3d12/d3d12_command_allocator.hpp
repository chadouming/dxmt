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
#include "util_env.hpp"
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
  std::vector<uint32_t> deps_;         // scratch
  std::vector<const void *> covered_; // scratch: resources whose older writers a dependency already covers
  // M2: resources leaving a write state, with the barrier call (1-based) that moved them, since the list's last join.
  std::vector<std::pair<const void *, uint32_t>> transitions_;
  EncoderData *clears_from_ = nullptr;        // M4: the encoder before the clears that end the list
  std::vector<ClearEncoderData *> clears_, kept_; // scratch

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
    if (encoder_current->type == EncoderType::Clear && encoder_last->type != EncoderType::Clear)
      clears_from_ = encoder_last;
    encoder_last->next = encoder_current;
    encoder_last = encoder_current;

    encoder_current = nullptr;
    encoder_count_++;
  }

  // M4 (GPU overlap spec §3.6): the clears that end the list, recorded since the last barrier call, whose view is an
  // attachment of render pass `r` (just opened) become that attachment's clear load action, a later clear of a view
  // winning. A Metal load action clears the render area only: the view must be its size. The other clears are
  // appended again, in order, ordered anew. Returns the number folded; none with DXMT_D3D12_MERGE=0 (triage).
  unsigned
  FoldClears(RenderEncoderData *r) {
    static const bool on = env::getEnvVar("DXMT_D3D12_MERGE") != "0";
    if (!on || encoder_last->type != EncoderType::Clear)
      return 0;
    auto *before = clears_from_;
    while (before->next->barriers != r->barriers) // a barrier after these clears: they stay
      if (!(before = before->next)->next)
        return 0;
    auto folds = [&](ClearEncoderData *c, bool apply) {
      if (c->width != r->render_target_width || c->height != r->render_target_height ||
          c->array_length != r->render_target_array_length)
        return false;
      if (c->clear_dsv) {
        if (((c->clear_dsv & 1) && r->depth.attachment.ptr() != c->attachment.ptr()) ||
            ((c->clear_dsv & 2) && r->stencil.attachment.ptr() != c->attachment.ptr()))
          return false;
        if (apply && (c->clear_dsv & 1)) {
          r->depth.load_action = WMTLoadActionClear;
          r->depth.clear_depth = c->depth_stencil.first;
        }
        if (apply && (c->clear_dsv & 2)) {
          r->stencil.load_action = WMTLoadActionClear;
          r->stencil.clear_stencil = c->depth_stencil.second;
        }
        return true;
      }
      for (auto &color : r->colors)
        if (color.attachment && color.attachment.ptr() == c->attachment.ptr() && color.depth_plane == c->depth_plane) {
          if (apply) {
            color.load_action = WMTLoadActionClear;
            color.clear_color = c->color;
          }
          return true;
        }
      return false;
    };
    clears_.clear();
    for (auto *e = before->next; e; e = e->next)
      clears_.push_back(static_cast<ClearEncoderData *>(e));
    // Newest first, the clears that stay (kept_, nulled in clears_): those that don't fold, and those a later clear
    // that stays writes the texture of (through another view), which would otherwise run before them.
    kept_.clear();
    for (auto it = clears_.rbegin(); it != clears_.rend(); ++it)
      if (!folds(*it, false) ||
          std::any_of(kept_.begin(), kept_.end(), [&](auto *k) { return k->writes[0] == (*it)->writes[0]; })) {
        kept_.push_back(*it);
        *it = nullptr;
      }
    if (kept_.size() == clears_.size())
      return 0;
    // Unlinked, they leave the list (and its group: only the first of them can have joined) as before them.
    before->next = nullptr;
    encoder_last = before;
    encoder_count_ -= clears_.size();
    group_.resize(group_.size() - clears_.size());
    unsigned folded = 0;
    for (auto *c : clears_)
      if (c) {
        folds(c, true);
        c->~ClearEncoderData(); // out of the list, Reset won't destroy it: its view reference goes now
        folded++;
      }
    for (auto it = kept_.rbegin(); it != kept_.rend(); ++it) {
      auto *c = *it;
      c->next = nullptr;
      c->deps = nullptr;
      c->dep_count = 0;
      encoder_current = c;
      InvalidateCurrentPass();
    }
    encoder_current = r;
    return folded;
  }

  HRESULT
  StartRecord(EncoderData **pStartEncoder) {
    if (encoder_last)
      return E_INVALIDARG;
    barriers_ = 0;
    joins_.assign(1, 0);
    group_.clear();
    transitions_.clear();
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

  // A barrier call: `join` when it orders all earlier work before all later work; otherwise the resources it moves
  // out of a write state, whose writers later work waits on (M2).
  void
  Barrier(bool join, const void *const *transitioned = nullptr, size_t count = 0) {
    barriers_++;
    joins_.push_back(joins_.back() + join);
    if (!join)
      for (size_t i = 0; i < count; i++)
        transitions_.push_back({transitioned[i], barriers_});
  }

  static bool
  Lists(const EncoderData *e, const void *resource) {
    for (unsigned i = 0; i < e->write_count; i++)
      if (e->writes[i] == resource)
        return true;
    return false;
  }

  // M2: `resource`, which g writes, left its write state after g began and before e's last command.
  bool
  Transitioned(const EncoderData *g, const EncoderData *e, const void *resource) {
    for (auto &[r, at] : transitions_)
      if (r == resource && at > g->barriers && at <= e->barriers_last)
        return true;
    return false;
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


  // What the closing encoder `e` waits on (GPU overlap spec §3.1). A join (all earlier work) when it is the list's
  // first, may write anything, or a joining barrier came after the previous encoder began and before e's last
  // command. Otherwise, of the encoders since the list's last join, the newest that writes each resource it writes
  // or that a barrier since moved out of its write state, and any that may write anything.
  void
  Decide(EncoderData *e) {
    auto prev = encoder_last; // the list's previous encoder, or its Null head
    e->position = encoder_count_;
    e->join = prev->type == EncoderType::Null || e->writes_unknown || joins_[e->barriers_last] > joins_[prev->barriers];
    if (e->join) {
      group_.clear();
      // Transitions before e began have their writers behind e, and so behind everything that waits after e.
      transitions_.erase(std::remove_if(transitions_.begin(), transitions_.end(),
                                        [&](auto &t) { return t.second <= e->barriers; }),
                         transitions_.end());
    } else {
      // Newest first, the newest writer of each resource only: an encoder taken has itself waited on the older
      // writers of everything it writes (GPU overlap spec §3.9).
      deps_.clear();
      covered_.clear();
      for (auto it = group_.rbegin(); it != group_.rend(); ++it) {
        auto *g = *it;
        bool needed = g->writes_unknown;
        for (unsigned i = 0; i < g->write_count && !needed; i++) {
          auto r = g->writes[i];
          needed = std::find(covered_.begin(), covered_.end(), r) == covered_.end() &&
                   (Lists(e, r) || Transitioned(g, e, r));
        }
        if (!needed)
          continue;
        deps_.push_back(g->position);
        covered_.insert(covered_.end(), g->writes, g->writes + g->write_count);
      }
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