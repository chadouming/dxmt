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

#include "com/com_guid.hpp"
#include "d3d12_stats.hpp"
#include "com/com_pointer.hpp"
#include "d3d12_device.hpp"
#include "d3d12_pageable.hpp"
#include "dxgi_interfaces.h"
#include "log/log.hpp"
#include "util_env.hpp"
#include "dxmt_capture.hpp"
#include "dxmt_format.hpp"
#include "d3d12_dxil_dump.hpp"
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <array>
#include <type_traits>
#include <vector>
#include "d3d10_1.h"
#include "d3d11_4.h"

namespace dxmt {

constexpr auto kCommandQueueSize = 32u;

const GUID kD3D12CommandQueueDownlevelUUID = {
    0x38a8c5ef, 0x7ccb, 0x4e81, {0x91, 0x4f, 0xa6, 0xe9, 0xd0, 0x72, 0xc4, 0x94}
};

// DXMT_STATS: names each Metal encoder "pass-<n> <kind>", n as in F9's passes.txt, for Instruments' encoder list.
// Render passes add their targets, "3c+d 1728x1120", to tell passes apart when their numbers shift between frames.
template <typename Encoder>
static void
LabelPass(Encoder &encoder, unsigned pass, const char *kind, RenderEncoderData *render = nullptr) {
  if (!g_stats_on)
    return;
  char name[64];
  int n = snprintf(name, sizeof(name), "pass-%u %s", pass, kind);
  if (render) {
    unsigned colors = 0;
    WMT::Texture first;
    for (auto &c : render->colors)
      if (c.attachment && colors++ == 0)
        first = c.attachment.texture();
    if (!first && render->depth.attachment)
      first = render->depth.attachment.texture();
    n += snprintf(name + n, sizeof(name) - n, " %uc%s", colors, render->depth.attachment ? "+d" : "");
    if (first)
      snprintf(name + n, sizeof(name) - n, " %llux%llu", (unsigned long long)first.width(),
               (unsigned long long)first.height());
  }
  encoder.setLabel(WMT::String::string(name, WMTUTF8StringEncoding));
}

class MTLD3D12CommandQueueImpl : public MTLD3D12Pageable<MTLD3D12CommandQueue, IMTLSwapChainFactory> {

  D3D12_COMMAND_QUEUE_DESC desc_;

  WMT::Reference<WMT::CommandQueue> queue_;
  // MacNeutron: encoder ordering (GPU overlap spec §3.3, §3.9). Each encoder updates a fence of its own from this ring
  // and waits on fences of earlier encoders: a join on every encoder since the last join (frontier_), any other
  // encoder on its group's fence (group_wait_: the join's early fence, updated after its first stage when it draws,
  // else the join's own fence) and on its dependencies in its command list. A fence is taken again only once no later
  // encoder can need its last update: an encoder that would take it sooner joins first.
  static constexpr unsigned kFences = 256;
  static constexpr uint16_t kNoFence = 0xffff;
  std::array<WMT::Reference<WMT::Fence>, kFences> fences_;
  std::array<uint64_t, kFences> fence_group_ = {}; // the join group that last updated each fence
  uint64_t group_ = 2;                             // fences of this group and the previous one are live
  uint16_t next_fence_ = 0;
  std::vector<uint16_t> frontier_, waits_;
  uint16_t group_wait_ = kNoFence, early_ = kNoFence; // early_: the early fence the last Order gave its join
  std::vector<uint16_t> list_fences_; // by position in the command list being encoded
  uint32_t list_join_ = 0;            // that list's last join, by position (UINT32_MAX: the queue's own work)
  // Strict order, every encoder a join, unless DXMT_D3D12_OVERLAP=1: on Apple GPUs overlap added more idle between
  // encoders than it saved (docs/testing/acceptance-dxmt-gpu-overlap.md, MacNeutron). Dumps and pixel history are
  // always strict; DXMT_D3D12_SERIAL=1 still forces it.
  bool serial_ = env::getEnvVar("DXMT_D3D12_OVERLAP") != "1" || env::getEnvVar("DXMT_D3D12_SERIAL") == "1";
  std::vector<wmtcmd_render_fence_op> render_ops_;
  std::vector<wmtcmd_blit_fence_op> blit_ops_;
  std::vector<wmtcmd_compute_fence_op> compute_ops_;

  // Takes the fence the next encoder updates and fills waits_ with the fences it waits on (and early_ with the early
  // fence a drawing join render pass also updates). `e` null: the queue's own work (pass dumps, pixel history,
  // presents), which joins.
  uint16_t
  Order(EncoderData *e, bool serial) {
    uint16_t fence = next_fence_;
    next_fence_ = (next_fence_ + 1) % kFences;
    bool join = !e || e->join || serial || fence_group_[fence] + 1 >= group_;
    waits_.clear();
    early_ = kNoFence;
    if (join) {
      waits_ = frontier_;
      frontier_.clear();
      group_++;
      list_join_ = e ? e->position : UINT32_MAX;
      group_wait_ = fence;
      if (!serial && e && e->type == EncoderType::Render && Draws(static_cast<RenderEncoderData *>(e))) {
        DXMT_STAT_COUNT("#encoder early fences", 1);
        early_ = group_wait_ = next_fence_; // live for the group; the join's own fence covers it for the next join
        next_fence_ = (next_fence_ + 1) % kFences;
        fence_group_[early_] = group_;
      }
    } else {
      waits_.assign(1, group_wait_);
      for (uint32_t i = 0; i < e->dep_count; i++)
        if (e->deps[i] >= list_join_) // earlier ones are behind the join, so behind the group fence
          waits_.push_back(list_fences_[e->deps[i]]);
    }
    if (e && g_stats_on)
      CountOrder(e, join);
    fence_group_[fence] = group_;
    frontier_.push_back(fence);
    if (e) {
      if (e->position >= list_fences_.size())
        list_fences_.resize(e->position + 1);
      list_fences_[e->position] = fence;
    }
    return fence;
  }

  // A draw among a render pass's commands: then its first stage, after its waits, proves them met (spec §3.9).
  static bool
  Draws(RenderEncoderData *data) {
    for (auto *cmd = (wmtcmd_base *)data->cmd_head.next.get(); cmd; cmd = (wmtcmd_base *)cmd->next.get())
      switch (cmd->type) {
      case WMTRenderCommandDraw:
      case WMTRenderCommandDrawIndexed:
      case WMTRenderCommandDrawIndirect:
      case WMTRenderCommandDrawIndexedIndirect:
      case WMTRenderCommandDrawMeshThreadgroups:
      case WMTRenderCommandDrawMeshThreadgroupsIndirect:
      case WMTRenderCommandDXMTGeometryDraw:
      case WMTRenderCommandDXMTGeometryDrawIndexed:
      case WMTRenderCommandDXMTGeometryDrawIndirect:
      case WMTRenderCommandDXMTGeometryDrawIndexedIndirect:
      case WMTRenderCommandDXMTTessellationMeshDraw:
      case WMTRenderCommandDXMTTessellationMeshDrawIndexed:
      case WMTRenderCommandDXMTTessellationMeshDrawIndirect:
      case WMTRenderCommandDXMTTessellationMeshDrawIndexedIndirect:
      case WMTRenderCommandDispatchThreadsPerTile:
      case WMTRenderCommandExecuteCommandsInBuffer:
        return true;
      default:
        break;
      }
    return false;
  }

  // M3 (GPU overlap spec §3.5): passes encodable as one Metal render pass: the same attachments, the later loading
  // them all and the earlier storing them all, with the same size, sample count, geometry and query heap.
  static bool
  SameTarget(RenderEncoderData *a, RenderEncoderData *b) {
    auto same = [](auto &x, auto &y) {
      return x.attachment.ptr() == y.attachment.ptr() && x.level == y.level && x.slice == y.slice &&
             x.depth_plane == y.depth_plane && (!y.attachment || y.load_action == WMTLoadActionLoad) &&
             (!x.attachment || x.store_action == WMTStoreActionStore);
    };
    for (unsigned i = 0; i < a->colors.size(); i++)
      if (!same(a->colors[i], b->colors[i]))
        return false;
    return same(a->depth, b->depth) && same(a->stencil, b->stencil) &&
           a->render_target_width == b->render_target_width && a->render_target_height == b->render_target_height &&
           a->render_target_array_length == b->render_target_array_length &&
           a->default_raster_sample_count == b->default_raster_sample_count &&
           a->dsv_planar_flags == b->dsv_planar_flags && a->dsv_readonly_flags == b->dsv_readonly_flags &&
           a->visibility_buffer == b->visibility_buffer && a->use_geometry == b->use_geometry;
  }

  bool merge_ = env::getEnvVar("DXMT_D3D12_MERGE") != "0"; // M3 on, unless DXMT_D3D12_MERGE=0 (triage)
  std::vector<EncoderData *> merged_, folds_;               // PlanMerge's result and scratch
  std::vector<std::pair<EncoderData *, uint16_t>> skipped_; // encoders M3 already encoded, with their fence
  size_t skip_ = 0;

  // M3: the render passes encoded inside `base`'s Metal render pass, and the timestamp-only blits whose samples move
  // to its end, from the encoders after base in this ExecuteCommandLists call (lists[list] is base's). A pass joins
  // when only list boundaries and timestamp-only blits come between, with no barrier, its targets are base's
  // (SameTarget), it resolves no indirect commands, and the samples fit (4 counter buffers, one sample each). Base
  // must be its group's join. Fills merged_ in order, and `info` with the moved samples.
  // ponytail: one ExecuteCommandLists call; merging across calls B coalesced would defer encoding to commit time.
  void
  PlanMerge(RenderEncoderData *base, unsigned list, ID3D12CommandList *const *lists, unsigned count,
            WMTRenderPassInfo &info) {
    merged_.clear();
    folds_.clear();
    if (!base->join || base->visibility_buffer) // a pass counting into a query: its counting would reach the next
      return;
    unsigned samples = base->num_samples;
    auto fits = [&](EncoderData *e, unsigned &n) {
      for (unsigned k = 0; k < e->num_samples; k++) {
        if (n == std::size(info.sample_buffers))
          return false;
        for (unsigned j = 0; j < n; j++)
          if (info.sample_buffers[j].sample_buffer == e->samples[k].buffer)
            return false;
        info.sample_buffers[n++] = {e->samples[k].buffer, e->samples[k].index};
      }
      return true;
    };
    EncoderData *last = base, *cursor = base->next;
    unsigned l = list, last_list = list;
    auto barriers = [&]() { return l == last_list ? last->barriers_last : 0u; }; // allowed before cursor, in list l
    for (;;) {
      if (!cursor) { // the end of list l: no barrier after `last` (or anywhere in a list of timestamps alone)
        if (static_cast<MTLD3D12GraphicsCommandList *>(lists[l])->barrier_count != barriers() || ++l == count ||
            std::find(lists + list, lists + l, lists[l]) != lists + l) // a list executed twice in this call
          break;
        cursor = static_cast<MTLD3D12GraphicsCommandList *>(lists[l])->entry;
        continue;
      }
      if (cursor->type == EncoderType::Null) {
        cursor = cursor->next;
        continue;
      }
      if (cursor->barriers_last != barriers()) // no barrier before it, or inside it, since `last`
        break;
      if (cursor->type == EncoderType::Blit && cursor->timestamp_only) {
        folds_.push_back(cursor);
        cursor = cursor->next;
        continue;
      }
      if (cursor->type != EncoderType::Render)
        break;
      auto *next = static_cast<RenderEncoderData *>(cursor);
      unsigned n = samples;
      bool ok = !next->pre_tail && SameTarget(base, next);
      for (auto *f : folds_)
        ok = ok && fits(f, n);
      ok = ok && fits(next, n);
      if (!ok) { // put back what `fits` wrote past the samples kept
        for (unsigned j = samples; j < n; j++)
          info.sample_buffers[j] = {};
        break;
      }
      samples = n;
      merged_.insert(merged_.end(), folds_.begin(), folds_.end());
      merged_.push_back(next);
      folds_.clear();
      last = next;
      last_list = l;
      cursor = next->next;
    }
  }

  void
  CountOrder(const EncoderData *e, bool join) { // DXMT_STATS (GPU overlap spec §3.8)
    static const unsigned joins = StatId("#encoder full joins"), listed = StatId("#encoders with a dependency list"),
                          dep_waits = StatId("#encoder dependency waits"),
                          fence_waits = StatId("#encoder fence waits"),
                          free = StatId("#encoder boundaries free to overlap");
    StatCount(fence_waits, waits_.size());
    if (join) {
      StatCount(joins);
      return;
    }
    size_t deps = waits_.size() - 1; // beyond the group fence
    if (deps) {
      StatCount(listed);
      StatCount(dep_waits, deps);
    }
    if (std::find(waits_.begin(), waits_.end(), list_fences_[e->position - 1]) == waits_.end())
      StatCount(free);
  }

  // A timestamp taken right after the earlier work, in a blit of its own (Metal drops an empty encoder: it fills 4
  // bytes of a scratch buffer). For counter buffers an encoder can't take beside another (GPU efficiency spec E2).
  WMT::Reference<WMT::Buffer> sample_scratch_;
  void
  SampleAfter(WMT::CommandBuffer &cmdbuf, obj_handle_t sample_buffer, uint64_t index) {
    DXMT_STAT_COUNT("#timestamps given their own encoder", 1);
    if (!sample_scratch_) {
      WMTBufferInfo info{};
      info.length = 16;
      info.options = WMTResourceOptions(WMTResourceStorageModePrivate | WMTResourceHazardTrackingModeUntracked);
      info.memory.set(nullptr);
      sample_scratch_ = device_->GetMTLDevice().newBuffer(info);
    }
    WMTSampleBufferAttachmentInfo attachment = {sample_buffer, ~0ull, index};
    auto blit = cmdbuf.blitCommandEncoderWithSampleBuffers(&attachment, 1);
    auto fence = Join(blit);
    wmtcmd_blit_fillbuffer fill = {};
    fill.type = WMTBlitCommandFillBuffer;
    fill.next.set(nullptr);
    fill.buffer = sample_scratch_.handle;
    fill.length = 4;
    blit.encodeCommands((const wmtcmd_blit_nop *)&fill);
    blit.updateFence(fences_[fence]);
    blit.endEncoding();
  }

  // The queue's own encoders (pass dumps, pixel history, presents) join. Returns the fence the encoder then updates.
  template <typename Encoder, typename... Stage>
  uint16_t
  Join(Encoder &encoder, Stage... before) {
    uint16_t fence = Order(nullptr, true);
    for (auto wait : waits_)
      encoder.waitForFence(fences_[wait], before...);
    return fence;
  }

  // The waits in waits_, an encoder's commands and the update of `fence` (and of `early`, after the `before` stages,
  // for a render pass) go to winemetal as one chained command list: each call crosses from Windows code to the Metal
  // side (several us under Rosetta). `head` null: the waits and the updates alone. The chain is unlinked again, as
  // pixel history re-encodes the commands. Render passes wait before `before` and update after the fragment stage.
  template <typename FenceOp, auto Wait, auto Update, typename Encoder, typename Nop>
  void
  EncodeOrdered(Encoder &encoder, std::vector<FenceOp> &ops, uint16_t fence, Nop *head, wmtcmd_base *tail,
                WMTRenderStages before = WMTRenderStageVertex, uint16_t early = kNoFence) {
    size_t n = waits_.size(), all = n + (early == kNoFence ? 1 : 2);
    ops.assign(all, FenceOp{});
    for (size_t i = 0; i < all; i++) {
      ops[i].type = i < n ? Wait : Update;
      ops[i].fence = fences_[i < n ? waits_[i] : i == n ? fence : early].handle;
      if constexpr (std::is_same_v<FenceOp, wmtcmd_render_fence_op>)
        ops[i].stages = i == n ? WMTRenderStageFragment : before;
      if (i + 1 < all && i + 1 != n)
        ops[i].next.set(&ops[i + 1]);
    }
    void *body = head ? (void *)head : (void *)&ops[n];
    if (n)
      ops[n - 1].next.set(body);
    if (tail)
      tail->next.set(&ops[n]);
    encoder.encodeCommands((const Nop *)(n ? (void *)ops.data() : body));
    if (tail)
      tail->next.set(nullptr);
  }

  void
  Encode(WMT::RenderCommandEncoder &encoder, uint16_t fence, WMTRenderStages before,
         wmtcmd_render_nop *head = nullptr, wmtcmd_base *tail = nullptr, uint16_t early = kNoFence) {
    EncodeOrdered<wmtcmd_render_fence_op, WMTRenderCommandWaitForFence, WMTRenderCommandUpdateFence>(
        encoder, render_ops_, fence, head, tail, before, early);
  }

  void
  Encode(WMT::BlitCommandEncoder &encoder, uint16_t fence, wmtcmd_blit_nop *head, wmtcmd_base *tail) {
    EncodeOrdered<wmtcmd_blit_fence_op, WMTBlitCommandWaitForFence, WMTBlitCommandUpdateFence>(encoder, blit_ops_,
                                                                                               fence, head, tail);
  }

  void
  Encode(WMT::ComputeCommandEncoder &encoder, uint16_t fence, wmtcmd_compute_nop *head, wmtcmd_base *tail) {
    EncodeOrdered<wmtcmd_compute_fence_op, WMTComputeCommandWaitForFence, WMTComputeCommandUpdateFence>(
        encoder, compute_ops_, fence, head, tail);
  }

  // Metal frame capture (MacNeutron; as DXMT's D3D11 queue: MTL_CAPTURE_ENABLED=1 and DXMT_CAPTURE_EXECUTABLE=<exe>,
  // then F10 or DXMT_CAPTURE_FRAME=<n>). Frames are counted by presents.
  CaptureState capture_state_;
  uint64_t presented_frames_ = 0;

  // Pass dump (MacNeutron capture mode, DXMT_DXIL_DUMP=<folder>): with DXMT_DUMP_FRAME=<n>, frame n's render pass
  // attachments, copied into shared buffers and saved as pass-<index>-<attachment>-<width>x<height>-<pixel format>.raw
  // eight frames later, with passes.txt listing every pass of the frame (and its queue). For finding which pass renders
  // wrong. DXMT_DUMP_FRAME=F9 dumps the frame after each press of F9 instead, over the previous dump. Shared by the
  // queues and counted by any queue's presents: Unreal presents through AMD's FSR 3 proxy, from the proxy's own queue.
  struct PassDump {
    WMT::Reference<WMT::Buffer> buffer;
    WMTBufferInfo info;
    std::string name;
  };
  struct PassDumps {
    std::mutex mutex;
    std::atomic_uint64_t frames = 0;
    std::atomic_uint64_t frame = ~0ull;
    uint64_t count = 1; // DXMT_DUMP_FRAMES: consecutive frames per dump (a one-frame glitch is hard to catch with F9)
    bool on_key = false;
    uint32_t saved = 0; // the last f9-<n> folder used; the next F9 takes the first free one after it
    uint32_t passes = 0, queues = 0;
    uint64_t bytes = 0;
    std::vector<PassDump> dumps;
    std::string log;
    // Pixel history (DXMT_DUMP_PIXEL=<x>,<y>[+<x>,<y>...][,<first pass>,<last pass>[,seq]]): each draw of the dumped frame's render
    // passes redrawn into scratch attachments, from the pass's starting state; pixels.txt lists the draws that change
    // the pixel, with their pipeline, and draws.txt every draw of the passes redrawn. Alone by default (a draw an
    // earlier draw would hide is listed too); with seq, each draw on top of the pass's earlier ones, as the pass draws
    // them (quadratic in the pass's draws). The redraws are full size: narrow the passes for a big frame.
    bool pixel_on = false, pixel_seq = false;
    std::string pixel_notes, pixel_draws;
    std::vector<std::pair<uint32_t, uint32_t>> pixels; // the pixels watched
    uint32_t pixel_first = 0, pixel_last = ~0u;
    struct PixelPass {
      std::string pass;
      bool seq = false;
      std::vector<std::pair<uint32_t, uint32_t>> points; // the watched pixels inside this pass
      std::vector<std::string> draws;       // each draw's pipeline
      std::vector<uint32_t> colors, texels; // the color attachments watched, and their texel sizes
      WMT::Reference<WMT::Buffer> buffer;   // 16 bytes per color: the pass without draws, then each draw alone
      WMTBufferInfo info;
    };
    std::vector<PixelPass> pixel_passes;
    std::vector<WMT::Reference<WMT::Texture>> pixel_textures; // scratch, kept until the dumped frame completes

    bool
    Dumping() { // the frame now encoding is one of the dumped ones
      uint64_t f = frame;
      return f != ~0ull && frames - f < count;
    }

    PassDumps() {
      auto value = env::getEnvVar("DXMT_DUMP_FRAME");
      if (value.empty() || !DXILCaptureMode())
        return;
      on_key = value == "F9";
      try {
        frame = std::stoull(value);
      } catch (const std::invalid_argument &) {
      }
      try {
        count = std::max<uint64_t>(1, std::stoull(env::getEnvVar("DXMT_DUMP_FRAMES")));
      } catch (const std::exception &) {
      }
      // Points joined by '+'; the pass range and mode follow the last one.
      std::string spec = env::getEnvVar("DXMT_DUMP_PIXEL");
      for (size_t start = 0; start < spec.size();) {
        size_t plus = spec.find('+', start);
        std::string token = spec.substr(start, plus == std::string::npos ? std::string::npos : plus - start);
        unsigned x, y, first, last;
        char mode[4] = {};
        int fields = sscanf(token.c_str(), "%u,%u,%u,%u,%3s", &x, &y, &first, &last, mode);
        if (fields < 2)
          break;
        pixels.push_back({x, y});
        if (fields >= 4) {
          pixel_first = first;
          pixel_last = last;
        }
        if (fields == 5)
          pixel_seq = !strcmp(mode, "seq");
        if (plus == std::string::npos)
          break;
        start = plus + 1;
      }
      pixel_on = !pixels.empty();
    }
  };
  static PassDumps &
  Dumps() {
    static PassDumps dumps;
    return dumps;
  }
  uint32_t queue_index_ = Dumps().queues++;

  void
  DumpAttachment(WMT::CommandBuffer &cmdbuf, const std::string &name, const TextureViewRef &view, uint32_t level,
                 uint32_t slice, uint32_t depth_plane, WMTBlitOption option) {
    auto &d = Dumps();
    auto texture = view.texture();
    auto format = texture.pixelFormat();
    uint64_t width = std::max<uint64_t>(1, texture.width() >> level), height = std::max<uint64_t>(1, texture.height() >> level);
    uint32_t texel = option ? 4 : MTLGetTexelSize(format);
    uint32_t samples = view->allocation && view->allocation->descriptor ? view->allocation->descriptor->sampleCount() : 1;
    std::string file = name + "-" + std::to_string(width) + "x" + std::to_string(height) + "-" + std::to_string(format);
    d.log += " " + file + (samples > 1 ? "-msaa" : "") + "@" + std::to_string(view->allocation ? (uint64_t)view->allocation->texture().handle : 0);
    uint64_t size = width * height * texel;
    // At most 3 GB of attachments per dump, 12 GB for a burst of frames (DXMT_DUMP_FRAMES), held in memory until saved.
    // ponytail: a fixed budget; save each frame as it completes if bursts need more.
    if (!texel || samples > 1 || d.bytes + size > (d.count > 1 ? 12ull << 30 : 3ull << 30))
      return;
    PassDump dump;
    dump.info.length = size;
    dump.info.options = WMTResourceStorageModeShared;
    dump.info.memory.set(0);
    dump.buffer = device_->GetMTLDevice().newBuffer(dump.info);
    if (!dump.buffer || !dump.info.memory.get_accessible_or_null())
      return;
    struct wmtcmd_blit_copy_from_texture_to_buffer_withblitoption cmd = {};
    cmd.type = WMTBlitCommandCopyFromTextureToBufferWithBlitOption;
    cmd.next.set(nullptr);
    cmd.src = texture.handle;
    cmd.slice = slice;
    cmd.level = level;
    cmd.origin = {0, 0, depth_plane};
    cmd.size = {width, height, 1};
    cmd.dst = dump.buffer.handle;
    cmd.offset = 0;
    cmd.bytes_per_row = width * texel;
    cmd.bytes_per_image = size;
    cmd.options = option;
    auto blit = cmdbuf.blitCommandEncoder();
    auto fence = Join(blit);
    blit.encodeCommands((const wmtcmd_blit_nop *)&cmd);
    blit.updateFence(fences_[fence]);
    blit.endEncoding();
    dump.name = file + ".raw";
    d.bytes += size;
    d.dumps.push_back(std::move(dump));
  }

  void
  DumpPass(WMT::CommandBuffer &cmdbuf, EncoderData *pass) {
    auto &d = Dumps();
    std::lock_guard<std::mutex> lock(d.mutex);
    std::string name = "pass-" + std::to_string(d.passes++);
    static const char *const kinds[] = {"null", "clear", "render", "blit", "compute", "resolve"};
    d.log += name + " q" + std::to_string(queue_index_) + " " + ((unsigned)pass->type < 6 ? kinds[(unsigned)pass->type] : "other");
    if (pass->type == EncoderType::Render) {
      auto data = static_cast<RenderEncoderData *>(pass);
      for (unsigned i = 0; i < data->colors.size(); i++) {
        auto &c = data->colors[i];
        if (c.attachment)
          DumpAttachment(cmdbuf, name + "-c" + std::to_string(i), c.attachment, c.level, c.slice, c.depth_plane, WMTBlitOptionNone);
      }
      if (data->depth.attachment) {
        auto format = data->depth.attachment.texture().pixelFormat();
        DumpAttachment(cmdbuf, name + "-d", data->depth.attachment, data->depth.level, data->depth.slice, 0,
                       format == WMTPixelFormatDepth32Float_Stencil8 ? WMTBlitOptionDepthFromDepthStencil : WMTBlitOptionNone);
      }
    }
    d.log += "\n";
  }

  // Pixel history for one render pass of the dumped frame (PassDumps). Encoded before the pass, into scratch textures
  // only, so the frame itself renders as it would have.
  void
  PixelHistory(WMT::CommandBuffer &cmdbuf, RenderEncoderData *data, const WMTRenderPassInfo &real) {
    auto &d = Dumps();
    std::lock_guard<std::mutex> lock(d.mutex);
    uint32_t pass = d.passes; // DumpPass numbers this pass once it is encoded
    if (!d.pixel_on || pass < d.pixel_first || pass > d.pixel_last)
      return;
    auto skipped = [&](const char *why) { d.pixel_notes += "# pass-" + std::to_string(pass) + " not redrawn: " + why + "\n"; };
    if (data->render_target_array_length > 1)
      return skipped("layered");
    // The draws, each under the pipeline set before it; and the visibility commands, which mustn't count into a
    // query heap on a redraw.
    std::vector<wmtcmd_base *> draws, visibility;
    std::vector<std::string> names;
    std::string pipeline = "(no pipeline)";
    for (auto *cmd = (wmtcmd_base *)&data->cmd_head; cmd; cmd = (wmtcmd_base *)cmd->next.get()) {
      switch (cmd->type) {
      case WMTRenderCommandSetPSO:
        pipeline = PipelineName(((wmtcmd_render_setpso *)cmd)->pso);
        break;
      case WMTRenderCommandSetVisibilityMode:
        visibility.push_back(cmd);
        break;
      case WMTRenderCommandDraw:
      case WMTRenderCommandDrawIndexed:
      case WMTRenderCommandDrawIndirect:
      case WMTRenderCommandDrawIndexedIndirect:
      case WMTRenderCommandDrawMeshThreadgroups:
      case WMTRenderCommandDrawMeshThreadgroupsIndirect:
      case WMTRenderCommandDXMTGeometryDraw:
      case WMTRenderCommandDXMTGeometryDrawIndexed:
      case WMTRenderCommandDXMTGeometryDrawIndirect:
      case WMTRenderCommandDXMTGeometryDrawIndexedIndirect:
      case WMTRenderCommandDXMTTessellationMeshDraw:
      case WMTRenderCommandDXMTTessellationMeshDrawIndexed:
      case WMTRenderCommandDXMTTessellationMeshDrawIndirect:
      case WMTRenderCommandDXMTTessellationMeshDrawIndexedIndirect:
      case WMTRenderCommandDispatchThreadsPerTile:
      case WMTRenderCommandExecuteCommandsInBuffer:
        draws.push_back(cmd);
        names.push_back(pipeline);
        break;
      default:
        break;
      }
    }
    if (draws.empty())
      return;
    for (size_t k = 0; k < names.size(); k++)
      d.pixel_draws += "pass-" + std::to_string(pass) + " draw-" + std::to_string(k) + " " + names[k] + "\n";

    auto metal = device_->GetMTLDevice();
    auto scratch = [&](WMTPixelFormat format, uint64_t width, uint64_t height) {
      WMTTextureInfo t = {};
      t.pixel_format = format;
      t.type = WMTTextureType2D;
      t.width = width;
      t.height = height;
      t.depth = 1;
      t.array_length = 1;
      t.mipmap_level_count = 1;
      t.sample_count = 1;
      t.usage = WMTTextureUsage(WMTTextureUsageRenderTarget | WMTTextureUsageShaderRead);
      t.options = WMTResourceOptions(WMTResourceStorageModePrivate);
      auto texture = metal.newTexture(t);
      d.pixel_textures.push_back(texture);
      return texture;
    };
    // The redraw's pass: scratch attachments, loaded (a DontCare load reads the pixel as it was) and stored.
    WMTRenderPassInfo info = real;
    info.visibility_buffer = 0;
    info.visibility_accumulate = false;
    for (auto &sample : info.sample_buffers)
      sample = {};
    struct Watched {
      WMT::Texture texture;
      uint32_t slice, level, plane;
      WMT::Reference<WMT::Texture> scratch, before; // before: one texel per watched pixel, in a row
      WMTPixelFormat format;
    };
    uint64_t width = ~0ull, height = ~0ull; // the smallest color attachment
    std::vector<Watched> colors;
    PassDumps::PixelPass record;
    record.pass = "pass-" + std::to_string(pass);
    record.draws = names;
    for (unsigned i = 0; i < std::size(info.colors); i++) {
      auto &c = data->colors[i];
      if (!c.attachment)
        continue;
      if (c.attachment->allocation && c.attachment->allocation->descriptor &&
          c.attachment->allocation->descriptor->sampleCount() > 1)
        return skipped("multisampled"); // ponytail: no pixel history in multisampled passes
      auto texture = c.attachment.texture();
      uint64_t w = std::max<uint64_t>(1, texture.width() >> c.level), h = std::max<uint64_t>(1, texture.height() >> c.level);
      width = std::min(width, w);
      height = std::min(height, h);
      auto format = texture.pixelFormat();
      colors.push_back({texture, c.slice, c.level, c.depth_plane, scratch(format, w, h), {}, format});
      auto &color = info.colors[i];
      color.texture = colors.back().scratch.handle;
      color.level = color.slice = color.depth_plane = 0;
      color.resolve_texture = 0;
      color.store_action = WMTStoreActionStore;
      if (color.load_action == WMTLoadActionDontCare)
        color.load_action = WMTLoadActionLoad;
      record.colors.push_back(i);
      record.texels.push_back(MTLGetTexelSize(format));
    }
    if (colors.empty())
      return skipped("no color attachment"); // ponytail: colors only
    for (auto &point : d.pixels)
      if (point.first < width && point.second < height)
        record.points.push_back(point);
    if (record.points.empty())
      return skipped("the pixels lie outside");
    for (auto &w : colors)
      w.before = scratch(w.format, record.points.size(), 1);
    // Depth and stencil (one texture in D3D12) are copied whole: Metal copies depth-stencil textures whole.
    WMT::Texture depth_texture = {};
    WMT::Reference<WMT::Texture> depth_scratch, depth_before;
    uint32_t depth_slice = 0, depth_level = 0;
    uint64_t depth_width = 0, depth_height = 0;
    if (data->stencil.attachment &&
        (!data->depth.attachment || data->stencil.attachment.texture().handle != data->depth.attachment.texture().handle))
      return skipped("separate stencil texture"); // ponytail: not redrawn
    if (data->depth.attachment) {
      depth_texture = data->depth.attachment.texture();
      depth_slice = data->depth.slice;
      depth_level = data->depth.level;
      depth_width = std::max<uint64_t>(1, depth_texture.width() >> depth_level);
      depth_height = std::max<uint64_t>(1, depth_texture.height() >> depth_level);
      depth_scratch = scratch(depth_texture.pixelFormat(), depth_width, depth_height);
      depth_before = scratch(depth_texture.pixelFormat(), depth_width, depth_height);
      auto redirect = [&](auto &plane) {
        if (!plane.texture)
          return;
        plane.texture = depth_scratch.handle;
        plane.level = plane.slice = plane.depth_plane = 0;
        plane.store_action = WMTStoreActionStore;
        if (plane.load_action == WMTLoadActionDontCare)
          plane.load_action = WMTLoadActionLoad;
      };
      redirect(info.depth);
      redirect(info.stencil);
    }
    // 16 bytes per color per point per redraw.
    const uint64_t slot = 16, stride = slot * colors.size() * record.points.size();
    record.info.length = stride * (draws.size() + 1);
    record.info.options = WMTResourceStorageModeShared;
    record.info.memory.set(0);
    record.buffer = metal.newBuffer(record.info);
    if (!record.buffer || !record.info.memory.get_accessible_or_null())
      return skipped("no readback buffer");
    record.seq = d.pixel_seq;

    auto copy = [&](WMT::BlitCommandEncoder &blit, obj_handle_t src, uint32_t slice, uint32_t level, WMTOrigin from,
                    WMTSize size, obj_handle_t dst, WMTOrigin to) {
      wmtcmd_blit_copy_from_texture_to_texture cmd = {};
      cmd.type = WMTBlitCommandCopyFromTextureToTexture;
      cmd.next.set(nullptr);
      cmd.src = src;
      cmd.src_slice = slice;
      cmd.src_level = level;
      cmd.src_origin = from;
      cmd.src_size = size;
      cmd.dst = dst;
      cmd.dst_origin = to;
      blit.encodeCommands((const wmtcmd_blit_nop *)&cmd);
    };
    auto read = [&](WMT::BlitCommandEncoder &blit, obj_handle_t src, uint32_t x, uint32_t y, uint32_t texel,
                    uint64_t offset) {
      wmtcmd_blit_copy_from_texture_to_buffer_withblitoption cmd = {};
      cmd.type = WMTBlitCommandCopyFromTextureToBufferWithBlitOption;
      cmd.next.set(nullptr);
      cmd.src = src;
      cmd.origin = {x, y, 0};
      cmd.size = {1, 1, 1};
      cmd.dst = record.buffer.handle;
      cmd.offset = offset;
      cmd.bytes_per_row = texel;
      cmd.bytes_per_image = texel;
      cmd.options = WMTBlitOptionNone;
      blit.encodeCommands((const wmtcmd_blit_nop *)&cmd);
    };
    // The pixels (and the whole depth) as they are before the pass.
    {
      auto blit = cmdbuf.blitCommandEncoder();
      auto before = Join(blit);
      for (auto &w : colors)
        for (uint32_t p = 0; p < record.points.size(); p++)
          copy(blit, w.texture.handle, w.slice, w.level, {record.points[p].first, record.points[p].second, w.plane},
               {1, 1, 1}, w.before.handle, {p, 0, 0});
      if (depth_before)
        copy(blit, depth_texture.handle, depth_slice, depth_level, {0, 0, 0}, {depth_width, depth_height, 1},
             depth_before.handle, {0, 0, 0});
      blit.updateFence(fences_[before]);
      blit.endEncoding();
    }
    std::vector<uint16_t> types;
    for (auto *cmd : draws)
      types.push_back(cmd->type);
    for (auto *cmd : visibility)
      cmd->type = WMTRenderCommandNop;
    // 0: no draw (the pass's clears alone); k: draw k - 1 alone, or draws 0 to k - 1 in sequence.
    for (size_t k = 0; k <= draws.size(); k++) {
      auto blit = cmdbuf.blitCommandEncoder();
      auto restored = Join(blit);
      for (auto &w : colors)
        for (uint32_t p = 0; p < record.points.size(); p++)
          copy(blit, w.before.handle, 0, 0, {p, 0, 0}, {1, 1, 1}, w.scratch.handle,
               {record.points[p].first, record.points[p].second, 0});
      if (depth_before)
        copy(blit, depth_before.handle, 0, 0, {0, 0, 0}, {depth_width, depth_height, 1}, depth_scratch.handle, {0, 0, 0});
      blit.updateFence(fences_[restored]);
      blit.endEncoding();
      for (size_t j = 0; j < draws.size(); j++)
        draws[j]->type = (record.seq ? j + 1 <= k : j + 1 == k) ? types[j] : (uint16_t)WMTRenderCommandNop;
      auto encoder = cmdbuf.renderCommandEncoder(info);
      auto drawn = Join(encoder, data->use_geometry ? WMTRenderStagePreRaster : WMTRenderStageVertex);
      encoder.encodeCommands(&data->cmd_head);
      encoder.updateFence(fences_[drawn], WMTRenderStageFragment);
      encoder.endEncoding();
      auto readback = cmdbuf.blitCommandEncoder();
      auto copied = Join(readback);
      for (size_t p = 0; p < record.points.size(); p++)
        for (size_t i = 0; i < colors.size(); i++)
          read(readback, colors[i].scratch.handle, record.points[p].first, record.points[p].second, record.texels[i],
               k * stride + (p * colors.size() + i) * slot);
      readback.updateFence(fences_[copied]);
      readback.endEncoding();
    }
    for (size_t j = 0; j < draws.size(); j++)
      draws[j]->type = types[j];
    for (auto *cmd : visibility)
      cmd->type = WMTRenderCommandSetVisibilityMode;
    d.pixel_notes += "# " + record.pass + ": " + std::to_string(draws.size()) + " draws redrawn" +
                     (record.seq ? " in sequence\n" : " alone\n");
    d.pixel_passes.push_back(std::move(record));
  }

  static void
  SaveDumps() {
    auto &d = Dumps();
    std::lock_guard<std::mutex> lock(d.mutex);
    if (d.log.empty())
      return;
    // F9 mode: each dump in its own new folder, f9-1, f9-2, ..., after any a previous run left.
    std::string dir;
    if (d.on_key) {
      do
        dir = "f9-" + std::to_string(++d.saved);
      while (!MakeCaptureFolder(dir.c_str()));
      dir += "\\";
    }
    for (auto &dump : d.dumps)
      SaveCapture((dir + dump.name).c_str(), dump.info.memory.get_accessible_or_null(), dump.info.length);
    if (d.pixel_on) {
      // "pass-<n> draw-<k> <pipeline> c<i> <before>-><after> ...": the draws that change the pixel. Before: the pass
      // without draws (alone), or after the draw before (in sequence).
      std::string text = d.pixel_notes;
      for (auto &p : d.pixel_passes) {
        auto *bytes = (const uint8_t *)p.info.memory.get_accessible_or_null();
        size_t colors = p.colors.size(), stride = 16 * colors * p.points.size();
        auto hex = [](const uint8_t *b, uint32_t n) {
          std::string s;
          char digits[3];
          for (uint32_t t = 0; t < n; t++) {
            snprintf(digits, sizeof(digits), "%02x", b[t]);
            s += digits;
          }
          return s;
        };
        for (size_t k = 1; k <= p.draws.size(); k++)
          for (size_t q = 0; q < p.points.size(); q++) {
            std::string changes;
            for (size_t i = 0; i < colors; i++) {
              size_t at = (q * colors + i) * 16;
              const uint8_t *without = bytes + (p.seq ? k - 1 : 0) * stride + at, *with = bytes + k * stride + at;
              if (memcmp(without, with, p.texels[i]))
                changes += " c" + std::to_string(p.colors[i]) + " " + hex(without, p.texels[i]) + "->" + hex(with, p.texels[i]);
            }
            if (!changes.empty())
              text += p.pass + " draw-" + std::to_string(k - 1) + " " + p.draws[k - 1] + changes + " at " +
                      std::to_string(p.points[q].first) + "," + std::to_string(p.points[q].second) + "\n";
          }
      }
      SaveCapture((dir + "pixels.txt").c_str(), text.data(), text.size());
      SaveCapture((dir + "draws.txt").c_str(), d.pixel_draws.data(), d.pixel_draws.size());
      d.pixel_notes.clear();
      d.pixel_draws.clear();
      d.pixel_passes.clear();
      d.pixel_textures.clear();
    }
    SaveCapture((dir + "passes.txt").c_str(), d.log.data(), d.log.size());
    d.dumps.clear();
    d.log.clear();
    d.passes = 0;
    d.bytes = 0;
    if (d.on_key)
      d.frame = ~0ull;
  }

  std::atomic_uint64_t inflight_cmdbuf_seq_ = 1;
  std::atomic_uint64_t inflight_cmdbuf_count_ = 0;
  std::atomic_uint64_t inflight_cmdbuf_stop_ = 0;

  // A fence signal deferred to the CPU (MacNeutron): to the generation it was asked of; `forward`: its MTLEvent signal
  // too (held back behind timestamps resolved into a custom heap, which the GPU can read).
  struct DeferredSignal {
    Rc<Fence> fence;
    Fence::Generation gen;
    uint64_t value;
    bool forward;
  };
  struct InflightCommandBuffer {
    WMT::Reference<WMT::CommandBuffer> cmdbuf{};
    HANDLE semaphore{};
    // Done on the CPU once it completes (MacNeutron): waits for the shared events of the fences its queue waited on
    // through their MTLEvents (GPU overlap spec §3.11), timestamp resolves, then the fence signals deferred behind
    // them, so no fence reports the work done before its timestamps are written, or before a fence it waited on.
    bool owes_cpu = false;
    std::vector<std::pair<WMT::Reference<WMT::SharedEvent>, uint64_t>> waits{};
    std::vector<TimestampResolve> resolves{};
    std::vector<DeferredSignal> signals{};
  };
  // Command buffers still owing CPU work: while there are any, fence signals go behind them.
  std::atomic_uint32_t cpu_work_ = 0;
  bool custom_resolves_ = false; // ponytail: sticky once a list resolved timestamps into a custom heap

  void
  Owe(InflightCommandBuffer &inflight) { // under mutex_commit_
    if (inflight.owes_cpu)
      return;
    inflight.owes_cpu = true;
    cpu_work_.fetch_add(1, std::memory_order_relaxed);
  }

  std::array<InflightCommandBuffer, kCommandQueueSize> inflight_cmdbuf_pool_;
  dxmt::thread inflight_cmdbuf_wait_thread_;

  dxmt::mutex mutex_commit_;

  void
  CommandBufferWaitingThread() {
    env::setThreadName("dxmt-cmdbuf-waiting-thread");
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);

    uint64_t internal_seq = 1;
    for (;;) {
      inflight_cmdbuf_seq_.wait(internal_seq, std::memory_order_acquire);
      if (inflight_cmdbuf_stop_.load() == internal_seq)
        break;
      auto &inflight = inflight_cmdbuf_pool_[internal_seq % kCommandQueueSize];

      if (inflight.cmdbuf.status() <= WMTCommandBufferStatusScheduled)
        inflight.cmdbuf.waitUntilCompleted();
      DXMT_STAT_SCOPE("queue.(retire a completed command buffer)");
      if (inflight.cmdbuf.status() == WMTCommandBufferStatusError)
        ERR("Device error: ", inflight.cmdbuf.error().description().getUTF8String());

      if (inflight.semaphore)
        ReleaseSemaphore(inflight.semaphore, 1, nullptr);

      if (inflight.owes_cpu) {
        // The GPU passed these MTLEvent waits, so each shared signal is encoded, being retired, or set (spec §3.11).
        // A failed buffer may not have: then its waits are skipped.
        if (inflight.cmdbuf.status() != WMTCommandBufferStatusError)
          for (auto &[shared, value] : inflight.waits)
            shared.waitUntilSignaledValue(value, ~0ull);
        for (auto &r : inflight.resolves) {
          MTLCounterSampleBuffer_resolveCounterRange(r.samples, r.start, r.count, r.dst, r.count * sizeof(uint64_t));
          for (auto [slot, sample] : r.aliases)
            MTLCounterSampleBuffer_resolveCounterRange(r.samples, sample, 1, (uint64_t *)r.dst + slot, sizeof(uint64_t));
        }
        // A failed buffer may not have run its MTLEvent signals: forwarded, every value reaches both events.
        bool failed = inflight.cmdbuf.status() == WMTCommandBufferStatusError;
        for (auto &d : inflight.signals) {
          d.gen.shared.signalValue(d.value);
          if (d.forward || failed)
            d.fence->forward(d.gen, d.value);
        }
        cpu_work_.fetch_sub(1, std::memory_order_release);
      }

      inflight = {};

      inflight_cmdbuf_count_.fetch_sub(1, std::memory_order_release);
      inflight_cmdbuf_count_.notify_one();

      internal_seq++;
    }
  };

  // MacNeutron (GPU overlap spec §3.10): an open Metal command buffer. A queue Wait encodes into it and waits for the
  // next call; ExecuteCommandLists, Signal and Present commit it, so the GPU starts on submitted work at once and
  // nothing ever waits on uncommitted work.
  bool open_ = false;

  // The open command buffer, after taking one of the 32 slots for a new one when none is open (under mutex_commit_).
  InflightCommandBuffer &
  Open() {
    auto &inflight = inflight_cmdbuf_pool_[inflight_cmdbuf_seq_.load(std::memory_order_relaxed) % kCommandQueueSize];
    if (open_)
      return inflight;
    {
      DXMT_STAT_SCOPE("queue.(wait for one of 32 command buffer slots)");
      inflight_cmdbuf_count_.wait(kCommandQueueSize, std::memory_order_acquire);
    }
    DXMT_STAT_SCOPE("queue.(new command buffer)");
    inflight.cmdbuf = queue_.commandBuffer();
    open_ = true;
    return inflight;
  }

  void
  Commit() { // under mutex_commit_
    if (!open_)
      return;
    DXMT_STAT_SCOPE("queue.(commit)");
    DXMT_STAT_COUNT("#command buffers committed", 1);
    inflight_cmdbuf_pool_[inflight_cmdbuf_seq_.load(std::memory_order_relaxed) % kCommandQueueSize].cmdbuf.commit();
    open_ = false;
    inflight_cmdbuf_seq_.fetch_add(1, std::memory_order_release);
    inflight_cmdbuf_seq_.notify_one();
    inflight_cmdbuf_count_.fetch_add(1, std::memory_order_relaxed);
  }

  struct CommittingScope {
    MTLD3D12CommandQueueImpl *queue;
    std::lock_guard<dxmt::mutex> lock;
    WMT::Reference<WMT::Object> pool;
    bool commit;
    InflightCommandBuffer &inflight;

    CommittingScope(MTLD3D12CommandQueueImpl *queue, bool commit) :
        queue(queue),
        lock(queue->mutex_commit_),
        pool(WMT::MakeAutoreleasePool()),
        commit(commit),
        inflight(queue->Open()) {}

    ~CommittingScope() {
      if (commit)
        queue->Commit();
    }
  };

  CommittingScope
  StartCommitting(bool commit) {
    return CommittingScope(this, commit);
  }

public:
  MTLD3D12CommandQueueImpl(MTLD3D12Device *pDevice) :
      MTLD3D12Pageable<MTLD3D12CommandQueue, IMTLSwapChainFactory>(pDevice),
      inflight_cmdbuf_wait_thread_([this]() { this->CommandBufferWaitingThread(); }) {
    auto frame = env::getEnvVar("DXMT_CAPTURE_FRAME");
    if (!frame.empty()) {
      try {
        capture_state_.scheduleNextFrameCapture(std::stoull(frame));
      } catch (const std::invalid_argument &) {
      }
    }
  }

  ~MTLD3D12CommandQueueImpl() {
    std::lock_guard<dxmt::mutex> lock(mutex_commit_);
    Commit(); // what ExecuteCommandLists left open
    inflight_cmdbuf_stop_.store(inflight_cmdbuf_seq_.fetch_add(1));
    inflight_cmdbuf_seq_.notify_one();
    inflight_cmdbuf_wait_thread_.join();
    SaveDumps(); // a dumped frame's passes completed: a test that never presents, or a game that quit
  }

  HRESULT
  Initialize(const D3D12_COMMAND_QUEUE_DESC *pDesc) {
    // TODO: validate and normalize
    desc_ = *pDesc;
    desc_.NodeMask = 1; // typically 1 GPU only

    auto metal_device = device_->GetMTLDevice();
    queue_ = metal_device.newCommandQueue(kCommandQueueSize);
    if (!queue_)
      return E_FAIL;
    queue_.addResidencySet(device_->GetGlobalResidencySet());

    for (auto &fence : fences_)
      fence = metal_device.newFence();

    return S_OK;
  }

  HRESULT
  STDMETHODCALLTYPE
  QueryInterface(REFIID riid, void **ppvObject) {
    DXMT_STAT_SCOPE("queue.QueryInterface");
    if (ppvObject == nullptr)
      return E_POINTER;

    *ppvObject = nullptr;

    if (riid == __uuidof(IUnknown) || riid == __uuidof(ID3D12Object) || riid == __uuidof(ID3D12DeviceChild) ||
        riid == __uuidof(ID3D12Pageable) || riid == __uuidof(ID3D12CommandQueue)) {
      *ppvObject = ref(this);
      return S_OK;
    }

    if (riid == __uuidof(IMTLSwapChainFactory)) {
      *ppvObject = ref_and_cast<IMTLSwapChainFactory>(this);
      return S_OK;
    }

    if (riid == __uuidof(ID3D10Device) || riid == __uuidof(ID3D10Device1))
      return E_NOINTERFACE;

    if (riid == __uuidof(ID3D11Device) || riid == __uuidof(ID3D11Device1) || riid == __uuidof(ID3D11Device2) ||
        riid == __uuidof(ID3D11Device3) || riid == __uuidof(ID3D11Device4) || riid == __uuidof(ID3D11Device5))
      return E_NOINTERFACE;

    if (riid == kD3D12CommandQueueDownlevelUUID)
      return E_NOINTERFACE;

    if (logQueryInterfaceError(__uuidof(ID3D12CommandQueue), riid)) {
      WARN("D3D12CommandQueue: Unknown interface query ", str::format(riid));
    }

    return E_NOINTERFACE;
  }

  void STDMETHODCALLTYPE UpdateTileMappings(
      ID3D12Resource *resource, UINT region_count, const D3D12_TILED_RESOURCE_COORDINATE *region_start_coordinates,
      const D3D12_TILE_REGION_SIZE *region_sizes, ID3D12Heap *heap, UINT range_count,
      const D3D12_TILE_RANGE_FLAGS *range_flags, const UINT *heap_range_offsets, const UINT *range_tile_counts,
      D3D12_TILE_MAPPING_FLAGS flags
  ) {
    DXMT_STAT_SCOPE("queue.UpdateTileMappings");
    IMPLEMENT_ME
  };

  void STDMETHODCALLTYPE CopyTileMappings(
      ID3D12Resource *dst_resource, const D3D12_TILED_RESOURCE_COORDINATE *dst_region_start_coordinate,
      ID3D12Resource *src_resource, const D3D12_TILED_RESOURCE_COORDINATE *src_region_start_coordinate,
      const D3D12_TILE_REGION_SIZE *region_size, D3D12_TILE_MAPPING_FLAGS flags
  ) {
    DXMT_STAT_SCOPE("queue.CopyTileMappings");
    IMPLEMENT_ME
  };

  void STDMETHODCALLTYPE
  ExecuteCommandLists(UINT Count, ID3D12CommandList *const *ppCommandLists) {
    DXMT_STAT_SCOPE("queue.ExecuteCommandLists");
    auto scope = StartCommitting(true); // commits at once: the GPU starts on it (a Wait before it rides along)
    auto &cmdbuf = scope.inflight.cmdbuf;
    bool dumping = Dumps().Dumping();
    bool serial = serial_ || dumping; // F9 dumps and pixel history see the strict order
    skipped_.clear();
    skip_ = 0;
    for (unsigned i = 0; i < Count; i++) {
      auto pCommandList = static_cast<MTLD3D12GraphicsCommandList *>(ppCommandLists[i]);
      custom_resolves_ |= pCommandList->custom_resolves;
      if (!pCommandList->timestamp_resolves.empty()) {
        Owe(scope.inflight);
        scope.inflight.resolves.insert(
            scope.inflight.resolves.end(), pCommandList->timestamp_resolves.begin(), pCommandList->timestamp_resolves.end()
        );
      }
      EncoderData *current = pCommandList->entry, *previous = nullptr;
      list_join_ = 0; // a list's first encoder joins, even one M3 already encoded
      while (current) {
        if (skip_ < skipped_.size() && skipped_[skip_].first == current) { // encoded inside an earlier render pass
          if (current->position >= list_fences_.size())
            list_fences_.resize(current->position + 1);
          list_fences_[current->position] = skipped_[skip_++].second;
          if (current->type == EncoderType::Render) // DXMT_STAT_COUNT keeps one counter id per call site
            DXMT_STAT_COUNT("#render passes merged", 1);
          else
            DXMT_STAT_COUNT("#timestamp blits folded", 1);
          current = current->next;
          continue;
        }
        unsigned pass = 0;
        if (g_stats_on) {
          static const unsigned kinds[] = {StatId("#null encoders"),   StatId("#clear passes"),
                                           StatId("#render passes"),   StatId("#blit passes"),
                                           StatId("#compute passes"),  StatId("#resolve passes")};
          static const unsigned boundaries = StatId("#encoder boundaries"),
                                no_barrier = StatId("#encoder boundaries with no barrier");
          pass = StatsNextPass();
          StatCount(kinds[(int)current->type]);
          if (current->type != EncoderType::Null) {
            if (previous) {
              StatCount(boundaries);
              if (current->barriers == previous->barriers)
                StatCount(no_barrier);
            }
            previous = current;
          }
        }
        switch (current->type) {
        case EncoderType::Null:
          break;
        case EncoderType::Clear: {
          DXMT_STAT_SCOPE("queue.(encode clear)");
          auto data = static_cast<ClearEncoderData *>(current);
          {
            WMTRenderPassInfo info;
            WMT::InitializeRenderPassInfo(info);
            if (data->clear_dsv) {
              if (data->clear_dsv & 1) {
                info.depth.clear_depth = data->depth_stencil.first;
                info.depth.texture = data->attachment.texture();
                info.depth.load_action = WMTLoadActionClear;
                info.depth.store_action = WMTStoreActionStore;
                info.depth.depth_plane = data->depth_plane;
              }
              if (data->clear_dsv & 2) {
                info.stencil.clear_stencil = data->depth_stencil.second;
                info.stencil.texture = data->attachment.texture();
                info.stencil.load_action = WMTLoadActionClear;
                info.stencil.store_action = WMTStoreActionStore;
                info.stencil.depth_plane = data->depth_plane;
              }
              info.render_target_width = data->width;
              info.render_target_height = data->height;
            } else {
              info.colors[0].clear_color = data->color;
              info.colors[0].texture = data->attachment.texture();
              info.colors[0].load_action = WMTLoadActionClear;
              info.colors[0].store_action = WMTStoreActionStore;
              info.colors[0].depth_plane = data->depth_plane;
            }
            info.render_target_array_length = data->array_length;
            auto encoder = cmdbuf.renderCommandEncoder(info);
            LabelPass(encoder, pass, "clear");
            Encode(encoder, Order(current, serial), WMTRenderStageFragment);
            encoder.endEncoding();
          }
          break;
        }
        case EncoderType::Render: {
          DXMT_STAT_SCOPE("queue.(encode render)");
          auto data = static_cast<RenderEncoderData *>(current);
          WMTRenderPassInfo render_pass_info;
          WMT::InitializeRenderPassInfo(render_pass_info);
          {
            for (unsigned i = 0; i < std::size(render_pass_info.colors); i++) {
              auto &color_data = data->colors[i];
              if (!color_data.attachment)
                continue;
              auto &color_info = render_pass_info.colors[i];
              color_info.texture = color_data.attachment.texture();
              color_info.load_action = color_data.load_action;
              color_info.store_action = color_data.store_action;
              color_info.level = color_data.level;
              color_info.slice = color_data.slice;
              color_info.depth_plane = color_data.depth_plane;
              color_info.clear_color = color_data.clear_color;
              color_info.resolve_texture = color_data.resolve_attachment.texture();
              color_info.resolve_level = color_data.resolve_level;
              color_info.resolve_slice = color_data.resolve_slice;
              color_info.resolve_depth_plane = color_data.resolve_depth_plane;
            }
            if (data->depth.attachment) {
              auto &depth_info = render_pass_info.depth;
              auto &depth_data = data->depth;
              depth_info.texture = depth_data.attachment.texture();
              depth_info.load_action = depth_data.load_action;
              depth_info.store_action = depth_data.store_action;
              depth_info.level = depth_data.level;
              depth_info.slice = depth_data.slice;
              depth_info.depth_plane = depth_data.depth_plane;
              depth_info.clear_depth = depth_data.clear_depth;
            }
            if (data->stencil.attachment) {
              auto &stencil_info = render_pass_info.stencil;
              auto &stencil_data = data->stencil;
              stencil_info.texture = stencil_data.attachment.texture();
              stencil_info.load_action = stencil_data.load_action;
              stencil_info.store_action = stencil_data.store_action;
              stencil_info.level = stencil_data.level;
              stencil_info.slice = stencil_data.slice;
              stencil_info.depth_plane = stencil_data.depth_plane;
              stencil_info.clear_stencil = stencil_data.clear_stencil;
            }
            render_pass_info.default_raster_sample_count = data->default_raster_sample_count;
            render_pass_info.render_target_array_length = data->render_target_array_length;
            render_pass_info.render_target_width = data->render_target_width;
            render_pass_info.render_target_height = data->render_target_height;
            render_pass_info.visibility_buffer = data->visibility_buffer;
            render_pass_info.visibility_accumulate = data->visibility_buffer != 0;
            for (unsigned i = 0; i < data->num_samples; i++)
              render_pass_info.sample_buffers[i] = {data->samples[i].buffer, data->samples[i].index};
          }
          if (!dumping && merge_) // M3: also in strict order, the default; dumps keep D3D12's passes
            PlanMerge(data, i, ppCommandLists, Count, render_pass_info);
          else
            merged_.clear();
          auto resolve_icbs = [&](uint16_t fence) { // ExecuteIndirect's resolvers, writing the pass's ICBs
            DXMT_STAT_COUNT("#indirect resolve passes", 1);
            auto pre = cmdbuf.computeCommandEncoder(true);
            LabelPass(pre, pass, "indirect resolve");
            Encode(pre, fence, &data->pre_head, data->pre_tail);
            pre.endEncoding();
          };
          if (dumping) { // strict order: the resolvers, the pass's pixel history, then the pass
            if (data->pre_tail)
              resolve_icbs(Order(nullptr, true));
            PixelHistory(cmdbuf, data, render_pass_info);
          }
          uint16_t fence = Order(current, serial), early = early_;
          if (data->pre_tail && !dumping) { // the resolvers wait as the pass would; the pass waits on them alone
            resolve_icbs(fence);
            waits_.assign(1, fence);
          }
          // Apple GPUs write only the last of an encoder's counter buffers (GPU efficiency spec E2): the pass keeps the
          // first; each other one's sample gets its own blit right after the pass.
          WMTRenderPassSampleBufferInfo trailing[std::size(render_pass_info.sample_buffers)] = {};
          for (unsigned i = 1; i < std::size(render_pass_info.sample_buffers); i++)
            std::swap(trailing[i], render_pass_info.sample_buffers[i]);
          auto encoder = cmdbuf.renderCommandEncoder(render_pass_info);
          LabelPass(encoder, pass, "render", data);
          // Geometry shader draws (MacNeutron) read resources in the object and mesh stages too.
          // Merged passes' commands follow this pass's, in one chain, unlinked again after.
          wmtcmd_base *tail = data->cmd_tail;
          for (auto *m : merged_)
            if (m->type == EncoderType::Render) {
              tail->next.set(&static_cast<RenderEncoderData *>(m)->cmd_head);
              tail = static_cast<RenderEncoderData *>(m)->cmd_tail;
            }
          Encode(encoder, fence, data->use_geometry ? WMTRenderStagePreRaster : WMTRenderStageVertex, &data->cmd_head,
                 tail, early);
          tail = data->cmd_tail;
          for (auto *m : merged_) {
            if (m->type == EncoderType::Render) {
              tail->next.set(nullptr);
              tail = static_cast<RenderEncoderData *>(m)->cmd_tail;
            }
            skipped_.push_back({m, fence});
          }
          encoder.endEncoding();
          for (auto &t : trailing)
            if (t.sample_buffer)
              SampleAfter(cmdbuf, t.sample_buffer, t.end_of_fragment_sample_index);
          break;
        }
        case EncoderType::Blit: {
          DXMT_STAT_SCOPE("queue.(encode blit)");
          auto data = static_cast<BlitEncoderData *>(current);
          WMTSampleBufferAttachmentInfo samples[4];
          for (unsigned i = 0; i < data->num_samples; i++)
            samples[i] = {data->samples[i].buffer, ~0ull, data->samples[i].index};
          auto encoder = data->num_samples ? cmdbuf.blitCommandEncoderWithSampleBuffers(samples, data->num_samples)
                                           : cmdbuf.blitCommandEncoder();
          LabelPass(encoder, pass, "blit");
          Encode(encoder, Order(current, serial), &data->cmd_head, data->cmd_tail);
          encoder.endEncoding();
          break;
        }
        case EncoderType::Compute: {
          DXMT_STAT_SCOPE("queue.(encode compute)");
          auto data = static_cast<ComputeEncoderData *>(current);
          WMTSampleBufferAttachmentInfo samples[4];
          for (unsigned i = 0; i < data->num_samples; i++)
            samples[i] = {data->samples[i].buffer, ~0ull, data->samples[i].index};
          auto encoder = data->num_samples ? cmdbuf.computeCommandEncoderWithSampleBuffers(false, samples, data->num_samples)
                                           : cmdbuf.computeCommandEncoder(false);
          LabelPass(encoder, pass, "compute");
          Encode(encoder, Order(current, serial), &data->cmd_head, data->cmd_tail);
          encoder.endEncoding();
          break;
        }
        case EncoderType::Resolve: {
          DXMT_STAT_SCOPE("queue.(encode resolve)");
          auto data = static_cast<ResolveEncoderData *>(current);

          WMTRenderPassInfo info;
          WMT::InitializeRenderPassInfo(info);
          info.colors[0].texture = data->src.texture();
          info.colors[0].load_action = WMTLoadActionLoad;
          info.colors[0].store_action = WMTStoreActionStoreAndMultisampleResolve;
          info.colors[0].resolve_texture = data->dst.texture();

          auto encoder = cmdbuf.renderCommandEncoder(info);
          encoder.setLabel(WMT::String::string("ResolvePass", WMTUTF8StringEncoding));
          Encode(encoder, Order(current, serial), WMTRenderStageFragment);
          encoder.endEncoding();

          break;
        }
        }
        if (dumping)
          DumpPass(cmdbuf, current);
        current = current->next;
      }
    }
  };

  void STDMETHODCALLTYPE SetMarker(UINT metadata, const void *data, UINT size) {
    DXMT_STAT_SCOPE("queue.SetMarker");};

  void STDMETHODCALLTYPE BeginEvent(UINT metadata, const void *data, UINT size) {
    DXMT_STAT_SCOPE("queue.BeginEvent");};

  void STDMETHODCALLTYPE EndEvent() {
    DXMT_STAT_SCOPE("queue.EndEvent");};

  HRESULT STDMETHODCALLTYPE
  Signal(ID3D12Fence *pFence, UINT64 Value) {
    DXMT_STAT_SCOPE("queue.Signal");
    auto scope = StartCommitting(true);
    auto &cmdbuf = scope.inflight.cmdbuf;
    auto &fence = static_cast<MTLD3D12Fence *>(pFence)->fence;
    auto gen = fence->ask(Value);
    // Other queues wait on the MTLEvent (GPU overlap spec §3.11): signaled at once, unless timestamps the GPU could
    // read (a custom heap's) are still to be written.
    bool owes = cpu_work_.load(std::memory_order_acquire) != 0, hold = owes && custom_resolves_;
    if (!hold)
      cmdbuf.encodeSignalEvent(gen.gpu, Value);
    if (owes) { // behind timestamps still to be written, or fences waited on: signal once they are
      DXMT_STAT_COUNT("#fence signals deferred to the CPU", 1);
      if (hold)
        DXMT_STAT_COUNT("#fence signals forwarded to the GPU event", 1);
      Owe(scope.inflight);
      scope.inflight.signals.push_back({fence, std::move(gen), Value, hold});
      return S_OK;
    }
    cmdbuf.encodeSignalEvent(gen.shared, Value);
    return S_OK;
  };

  HRESULT STDMETHODCALLTYPE
  Wait(ID3D12Fence *pFence, UINT64 Value) {
    DXMT_STAT_SCOPE("queue.Wait");
    auto gen = static_cast<MTLD3D12Fence *>(pFence)->fence->current();
    if (gen.shared.signaledValue() >= Value) {
      DXMT_STAT_COUNT("#queue waits already met", 1);
      return S_OK;
    }
    // On the MTLEvent (GPU overlap spec §3.11); the shared event's value is then owed before this queue's later
    // signals reach the CPU.
    DXMT_STAT_COUNT("#queue waits on the GPU event", 1);
    auto scope = StartCommitting(false);
    scope.inflight.cmdbuf.encodeWaitForEvent(gen.gpu, Value);
    Owe(scope.inflight);
    scope.inflight.waits.push_back({std::move(gen.shared), Value});
    return S_OK;
  };

  HRESULT STDMETHODCALLTYPE
  GetTimestampFrequency(UINT64 *pFrequency) {
    DXMT_STAT_SCOPE("queue.GetTimestampFrequency");
    if (!pFrequency)
      return E_INVALIDARG;
    *pFrequency = device_->TimestampFrequency();
    return S_OK;
  };

  HRESULT STDMETHODCALLTYPE
  GetClockCalibration(UINT64 *gpu_timestamp, UINT64 *cpu_timestamp) {
    DXMT_STAT_SCOPE("queue.GetClockCalibration");
    // Metal's GPU clock with QueryPerformanceCounter read on either side (Unreal treats a failure here as fatal).
    LARGE_INTEGER before, after;
    uint64_t cpu, gpu;
    QueryPerformanceCounter(&before);
    device_->GetMTLDevice().sampleTimestamps(cpu, gpu);
    QueryPerformanceCounter(&after);
    if (gpu_timestamp)
      *gpu_timestamp = gpu;
    if (cpu_timestamp)
      *cpu_timestamp = before.QuadPart + (after.QuadPart - before.QuadPart) / 2;
    return S_OK;
  };

  D3D12_COMMAND_QUEUE_DESC *STDMETHODCALLTYPE
  GetDesc(D3D12_COMMAND_QUEUE_DESC *__ret) {
    DXMT_STAT_SCOPE("queue.GetDesc");
    *__ret = desc_;
    return __ret;
  };

  HRESULT STDMETHODCALLTYPE
  CreateSwapChain(
      IDXGIFactory1 *pFactory, HWND hWnd, const DXGI_SWAP_CHAIN_DESC1 *pDesc,
      const DXGI_SWAP_CHAIN_FULLSCREEN_DESC *pFullscreenDesc, IDXGISwapChain1 **ppSwapChain
  ) {
    DXMT_STAT_SCOPE("queue.CreateSwapChain");
    return dxmt::CreateSwapChain(pFactory, device_, this, hWnd, pDesc, pFullscreenDesc, ppSwapChain);
  }

  HRESULT
  Present(Presenter *presenter, ID3D12Resource *backbuffer, HANDLE hLantecyWaitable, double after) {
    HRESULT hr = PresentFrame(presenter, backbuffer, hLantecyWaitable, after);
    // Once the present is committed: start or stop a frame capture at this boundary.
    auto &dumps = Dumps();
    if (dumps.Dumping()) { // the end of a dumped frame, in passes.txt
      std::lock_guard<std::mutex> lock(dumps.mutex);
      dumps.log += "# frame " + std::to_string(dumps.frames - dumps.frame) + " presented\n";
    }
    uint64_t frames = ++dumps.frames;
    StatsFrame();
    if (dumps.frame != ~0ull && frames == dumps.frame + dumps.count - 1 + 8)
      SaveDumps();
    else if (dumps.on_key && dumps.frame == ~0ull && (GetAsyncKeyState(VK_F9) & 0x8000))
      dumps.frame = frames; // the frame now starting
    switch (capture_state_.getNextAction(++presented_frames_)) {
    case CaptureState::NextAction::StartCapture: {
      WMTCaptureInfo info;
      info.capture_object = device_->GetMTLDevice();
      info.destination = WMTCaptureDestinationGPUTraceDocument;
      char stamp[64];
      std::time_t now = std::time(nullptr);
      std::strftime(stamp, sizeof(stamp), "_%H'%M'%S_%m-%d-%y.gputrace", std::localtime(&now));
      auto file = env::getUnixPath(env::getExeBaseName() + "_F." + std::to_string(presented_frames_ + 1) + stamp);
      WARN("A new capture will be saved to ", file);
      info.output_url.set(file.c_str());
      WMT::CaptureManager::sharedCaptureManager().startCapture(info);
      break;
    }
    case CaptureState::NextAction::StopCapture:
      WMT::CaptureManager::sharedCaptureManager().stopCapture();
      break;
    case CaptureState::NextAction::Nothing:
      if (capture_state_.shouldCaptureNextFrame())
        capture_state_.scheduleNextFrameCapture(presented_frames_ + 1);
      break;
    }
    return hr;
  }

  HRESULT
  PresentFrame(Presenter *presenter, ID3D12Resource *backbuffer, HANDLE hLantecyWaitable, double after) {
    auto scope = StartCommitting(true);
    auto &cmdbuf = scope.inflight.cmdbuf;

    auto g = reinterpret_cast<MTLD3D12Resource *>(backbuffer);
    auto &view = g->texture->view(g->texture->fullView);

    auto state = presenter->synchronizeLayerProperties();
    uint16_t fence = 0;
    auto drawable = presenter->encodeCommands(
        cmdbuf, view.texture, state.metadata,
        [&](auto encoder) { fence = Join(encoder, WMTRenderStageFragment); },
        [&](auto encoder) { encoder.updateFence(fences_[fence], WMTRenderStageFragment); }
    );

    if (after > 0)
      cmdbuf.presentDrawableAfterMinimumDuration(drawable, after);
    else
      cmdbuf.presentDrawable(drawable);
    scope.inflight.semaphore = hLantecyWaitable;

    return S_OK;
  }
};

HRESULT
CreateCommandQueue(MTLD3D12Device *pDevice, const D3D12_COMMAND_QUEUE_DESC *pDesc, REFIID riid, void **ppCommandQueue) {
  auto command_queue = Com(new MTLD3D12CommandQueueImpl(pDevice));
  HRESULT hr = command_queue->Initialize(pDesc);
  if (FAILED(hr))
    return hr;
  return command_queue->QueryInterface(riid, ppCommandQueue);
};

} // namespace dxmt