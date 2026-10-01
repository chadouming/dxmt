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

#pragma once
#include "d3d12.h"
#include "sha1/sha1_util.hpp"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// A `.pipelines` recording (shader pre-caching spec §3.5), what Fossilize's .foz files are to Steam's Vulkan
// pre-caching: "DXMTPRC1", then records of a 36-byte header (uint32 kind, uint32 payload size, uint64 FNV-1a of the
// payload, 20-byte SHA-1 id) and the payload. Blobs (shaders, root signatures) are stored once; pipeline payloads name
// them by id, all zeros for none.
namespace dxmt::record {

constexpr char kMagic[8] = {'D', 'X', 'M', 'T', 'P', 'R', 'C', '1'};
constexpr size_t kHeaderSize = 36;
enum Kind : uint32_t { kBlob = 1, kGraphics = 2, kCompute = 3 };

// A blob a pipeline refers to. Null data: none.
struct Blob {
  Sha1Digest id = {};
  const void *data = nullptr;
  size_t size = 0;
};

uint64_t Checksum(const void *data, size_t size);

std::vector<uint8_t> SerializeGraphics(const D3D12_GRAPHICS_PIPELINE_STATE_DESC &desc, const Sha1Digest &root,
                                       const Sha1Digest &vs, const Sha1Digest &ps, const Sha1Digest &gs);
std::vector<uint8_t> SerializeCompute(const D3D12_COMPUTE_PIPELINE_STATE_DESC &desc, const Sha1Digest &root,
                                      const Sha1Digest &cs);

// A pipeline read back. Its pointers are null: the replayer points them at blobs, and the input layout at `elements`
// (whose SemanticName point into `names`) just before creating it.
struct GraphicsRecord {
  Sha1Digest root, vs, ps, gs;
  D3D12_GRAPHICS_PIPELINE_STATE_DESC desc;
  std::vector<std::string> names;
  std::vector<D3D12_INPUT_ELEMENT_DESC> elements;
};
struct ComputeRecord {
  Sha1Digest root, cs;
  D3D12_COMPUTE_PIPELINE_STATE_DESC desc;
};
bool ParseGraphics(const std::vector<uint8_t> &payload, GraphicsRecord &record);
bool ParseCompute(const std::vector<uint8_t> &payload, ComputeRecord &record);

// A whole recording as the replayer reads it: every record's size and checksum verified.
struct Recording {
  bool opened = false; // false: missing, unreadable, or not a recording
  std::unordered_map<Sha1Digest, std::vector<uint8_t>> blobs;
  std::vector<std::pair<Kind, std::vector<uint8_t>>> pipelines;
  uint64_t bad = 0; // records skipped: cut short, unknown kind or wrong checksum
};
Recording Read(const std::wstring &path);

// The recorder: after a pipeline's creation succeeded, appends what `<DXMT_PIPELINE_RECORD>/<exe>.pipelines` lacks.
bool RecordingOn();
void RecordGraphics(const D3D12_GRAPHICS_PIPELINE_STATE_DESC &desc, const Blob &root, const Blob &vs, const Blob &ps,
                    const Blob &gs);
void RecordCompute(const D3D12_COMPUTE_PIPELINE_STATE_DESC &desc, const Blob &root, const Blob &cs);

} // namespace dxmt::record
