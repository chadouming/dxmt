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

#include "d3d12_pipeline_record.hpp"
#include "log/log.hpp"
#include "util_env.hpp"
#include "util_string.hpp"
#include <windows.h>
#include <cstring>
#include <mutex>
#include <type_traits>
#include <unordered_set>

namespace dxmt::record {

namespace {

const Sha1Digest kNone = {};

// Fields go one by one, never as whole structs, so padding never reaches a file or an id. Out writes a description,
// In reads one back; GraphicsFields and ComputeFields list the fields once for both.
struct Out {
  std::vector<uint8_t> bytes;
  void raw(const void *data, size_t size) {
    auto *p = static_cast<const uint8_t *>(data);
    bytes.insert(bytes.end(), p, p + size);
  }
  template <typename T> void field(T &value) {
    static_assert(std::is_arithmetic_v<T> || std::is_enum_v<T>);
    raw(&value, sizeof(T));
  }
  void field(Sha1Digest &digest) { raw(digest.data, sizeof(digest.data)); }
  void layout(D3D12_INPUT_LAYOUT_DESC &layout, std::vector<std::string> &, std::vector<D3D12_INPUT_ELEMENT_DESC> &);
};

struct In {
  const uint8_t *p, *end;
  bool ok = true;
  void raw(void *data, size_t size) {
    if (!ok || size_t(end - p) < size) {
      ok = false;
      memset(data, 0, size);
      return;
    }
    memcpy(data, p, size);
    p += size;
  }
  template <typename T> void field(T &value) {
    static_assert(std::is_arithmetic_v<T> || std::is_enum_v<T>);
    raw(&value, sizeof(T));
  }
  void field(Sha1Digest &digest) { raw(digest.data, sizeof(digest.data)); }
  void layout(D3D12_INPUT_LAYOUT_DESC &layout, std::vector<std::string> &names,
              std::vector<D3D12_INPUT_ELEMENT_DESC> &elements);
};

template <typename IO>
void
ElementFields(IO &io, D3D12_INPUT_ELEMENT_DESC &e) {
  io.field(e.SemanticIndex);
  io.field(e.Format);
  io.field(e.InputSlot);
  io.field(e.AlignedByteOffset);
  io.field(e.InputSlotClass);
  io.field(e.InstanceDataStepRate);
}

void
Out::layout(D3D12_INPUT_LAYOUT_DESC &layout, std::vector<std::string> &, std::vector<D3D12_INPUT_ELEMENT_DESC> &) {
  field(layout.NumElements);
  for (UINT i = 0; i < layout.NumElements; i++) {
    D3D12_INPUT_ELEMENT_DESC e = layout.pInputElementDescs[i];
    uint32_t length = e.SemanticName ? (uint32_t)strlen(e.SemanticName) : 0;
    field(length);
    raw(e.SemanticName, length);
    ElementFields(*this, e);
  }
}

void
In::layout(D3D12_INPUT_LAYOUT_DESC &layout, std::vector<std::string> &names,
           std::vector<D3D12_INPUT_ELEMENT_DESC> &elements) {
  field(layout.NumElements);
  if (layout.NumElements > D3D12_IA_VERTEX_INPUT_STRUCTURE_ELEMENT_COUNT)
    ok = false;
  for (UINT i = 0; ok && i < layout.NumElements; i++) {
    uint32_t length = 0;
    field(length);
    if (!ok || length > 256 || size_t(end - p) < length) {
      ok = false;
      break;
    }
    names.emplace_back(reinterpret_cast<const char *>(p), length);
    p += length;
    D3D12_INPUT_ELEMENT_DESC e = {};
    ElementFields(*this, e);
    elements.push_back(e);
  }
  layout.pInputElementDescs = nullptr;
}

template <typename IO>
void
GraphicsFields(IO &io, D3D12_GRAPHICS_PIPELINE_STATE_DESC &d, Sha1Digest (&ids)[4], std::vector<std::string> &names,
               std::vector<D3D12_INPUT_ELEMENT_DESC> &elements) {
  for (auto &id : ids) // root signature, VS, PS, GS
    io.field(id);
  auto &blend = d.BlendState;
  io.field(blend.AlphaToCoverageEnable);
  io.field(blend.IndependentBlendEnable);
  for (auto &rt : blend.RenderTarget) {
    io.field(rt.BlendEnable);
    io.field(rt.LogicOpEnable);
    io.field(rt.SrcBlend);
    io.field(rt.DestBlend);
    io.field(rt.BlendOp);
    io.field(rt.SrcBlendAlpha);
    io.field(rt.DestBlendAlpha);
    io.field(rt.BlendOpAlpha);
    io.field(rt.LogicOp);
    io.field(rt.RenderTargetWriteMask);
  }
  io.field(d.SampleMask);
  auto &r = d.RasterizerState;
  io.field(r.FillMode);
  io.field(r.CullMode);
  io.field(r.FrontCounterClockwise);
  io.field(r.DepthBias);
  io.field(r.DepthBiasClamp);
  io.field(r.SlopeScaledDepthBias);
  io.field(r.DepthClipEnable);
  io.field(r.MultisampleEnable);
  io.field(r.AntialiasedLineEnable);
  io.field(r.ForcedSampleCount);
  io.field(r.ConservativeRaster);
  auto &z = d.DepthStencilState;
  io.field(z.DepthEnable);
  io.field(z.DepthWriteMask);
  io.field(z.DepthFunc);
  io.field(z.StencilEnable);
  io.field(z.StencilReadMask);
  io.field(z.StencilWriteMask);
  for (auto *face : {&z.FrontFace, &z.BackFace}) {
    io.field(face->StencilFailOp);
    io.field(face->StencilDepthFailOp);
    io.field(face->StencilPassOp);
    io.field(face->StencilFunc);
  }
  io.layout(d.InputLayout, names, elements);
  io.field(d.IBStripCutValue);
  io.field(d.PrimitiveTopologyType);
  io.field(d.NumRenderTargets);
  for (auto &format : d.RTVFormats)
    io.field(format);
  io.field(d.DSVFormat);
  io.field(d.SampleDesc.Count);
  io.field(d.SampleDesc.Quality);
  io.field(d.NodeMask);
  io.field(d.Flags);
}

template <typename IO>
void
ComputeFields(IO &io, D3D12_COMPUTE_PIPELINE_STATE_DESC &d, Sha1Digest (&ids)[2]) {
  for (auto &id : ids) // root signature, CS
    io.field(id);
  io.field(d.NodeMask);
  io.field(d.Flags);
}

void
AppendRecord(std::vector<uint8_t> &out, Kind kind, const Sha1Digest &id, const void *payload, size_t size) {
  uint32_t k = kind, n = (uint32_t)size;
  uint64_t sum = Checksum(payload, size);
  auto put = [&](const void *data, size_t length) {
    auto *b = static_cast<const uint8_t *>(data);
    out.insert(out.end(), b, b + length);
  };
  put(&k, 4);
  put(&n, 4);
  put(&sum, 8);
  put(id.data, 20);
  put(payload, size);
}

bool
ReadAll(HANDLE file, std::vector<uint8_t> &bytes) {
  LARGE_INTEGER size;
  if (!GetFileSizeEx(file, &size) || size.QuadPart > 0x7fffffff)
    return false;
  bytes.resize((size_t)size.QuadPart);
  DWORD got = 0;
  return bytes.empty() || (ReadFile(file, bytes.data(), (DWORD)bytes.size(), &got, nullptr) && got == bytes.size());
}

bool
WriteAll(HANDLE file, const std::vector<uint8_t> &bytes) {
  DWORD written = 0;
  return WriteFile(file, bytes.data(), (DWORD)bytes.size(), &written, nullptr) && written == bytes.size();
}

struct Recorder {
  // ponytail: one lock and one write per new pipeline; a writer thread if recording shows in first-session profiles.
  std::mutex mutex;
  HANDLE file = INVALID_HANDLE_VALUE;
  bool opened = false, off = false;
  std::unordered_set<Sha1Digest> ids;
};

Recorder &
State() {
  static Recorder recorder;
  return recorder;
}

// `<DXMT_PIPELINE_RECORD>\<exe>.pipelines`, or empty when recording is off.
const std::wstring &
RecordingPath() {
  static const std::wstring path = [] {
    std::string folder = env::getEnvVar("DXMT_PIPELINE_RECORD");
    if (folder.empty() || folder == "0")
      return std::wstring();
    while (folder.size() > 1 && (folder.back() == '/' || folder.back() == '\\'))
      folder.pop_back();
    if (folder[0] == '/')
      folder = "Z:" + folder; // the launcher passes a Mac path; Wine's Z: drive is the Mac's root
    std::wstring wide = str::tows(folder.c_str());
    CreateDirectoryW(wide.c_str(), nullptr); // may exist; any other failure shows when the file doesn't open
    return wide + L"\\" + str::tows(env::getCacheExeName().c_str()) + L".pipelines";
  }();
  return path;
}

void
TurnOff(Recorder &r, const char *reason) {
  WARN("d3d12 pipeline recording off: ", reason);
  r.off = true;
  if (r.file != INVALID_HANDLE_VALUE) {
    CloseHandle(r.file);
    r.file = INVALID_HANDLE_VALUE;
  }
}

// Opens the recording once: reads its record ids and cuts a torn tail. Sizes only, no checksums: a game doesn't hash
// a large file while it loads.
bool
Ready(Recorder &r) {
  if (r.off)
    return false;
  if (r.opened)
    return true;
  r.opened = true;
  r.file = CreateFileW(RecordingPath().c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_ALWAYS,
                       FILE_ATTRIBUTE_NORMAL, nullptr);
  if (r.file == INVALID_HANDLE_VALUE) {
    TurnOff(r, "can't open the recording");
    return false;
  }
  std::vector<uint8_t> bytes;
  if (!ReadAll(r.file, bytes)) {
    TurnOff(r, "can't read the recording");
    return false;
  }
  size_t end = 0;
  if (bytes.size() >= sizeof(kMagic) && !memcmp(bytes.data(), kMagic, sizeof(kMagic))) {
    end = sizeof(kMagic);
    while (bytes.size() - end >= kHeaderSize) {
      uint32_t kind, size;
      Sha1Digest id;
      memcpy(&kind, bytes.data() + end, 4);
      memcpy(&size, bytes.data() + end + 4, 4);
      memcpy(id.data, bytes.data() + end + 16, 20);
      if (kind < kBlob || kind > kCompute || size > bytes.size() - end - kHeaderSize)
        break;
      r.ids.insert(id);
      end += kHeaderSize + size;
    }
  }
  // A foreign file starts over; anything after the last complete record is cut.
  LARGE_INTEGER at;
  at.QuadPart = (LONGLONG)end;
  if (!SetFilePointerEx(r.file, at, nullptr, FILE_BEGIN) || !SetEndOfFile(r.file)) {
    TurnOff(r, "can't cut the recording's torn tail");
    return false;
  }
  if (!end && !WriteAll(r.file, std::vector<uint8_t>(kMagic, kMagic + sizeof(kMagic)))) {
    TurnOff(r, "can't write the recording");
    return false;
  }
  return true;
}

void
Append(const std::vector<uint8_t> &payload, Kind kind, std::initializer_list<const Blob *> blobs) {
  Sha1Digest id = Sha1HashState::compute(payload.data(), payload.size());
  auto &r = State();
  std::lock_guard<std::mutex> lock(r.mutex);
  if (!Ready(r) || r.ids.count(id))
    return;
  std::vector<uint8_t> out;
  for (auto *blob : blobs)
    if (blob->data && r.ids.insert(blob->id).second)
      AppendRecord(out, kBlob, blob->id, blob->data, blob->size);
  AppendRecord(out, kind, id, payload.data(), payload.size());
  if (!WriteAll(r.file, out)) {
    TurnOff(r, "can't write the recording");
    return;
  }
  r.ids.insert(id);
}

Sha1Digest
IdOf(const Blob &blob) {
  return blob.data ? blob.id : kNone;
}

} // namespace

uint64_t
Checksum(const void *data, size_t size) {
  auto *p = static_cast<const uint8_t *>(data);
  uint64_t hash = 0xcbf29ce484222325ull; // FNV-1a
  for (size_t i = 0; i < size; i++)
    hash = (hash ^ p[i]) * 0x100000001b3ull;
  return hash;
}

std::vector<uint8_t>
SerializeGraphics(const D3D12_GRAPHICS_PIPELINE_STATE_DESC &desc, const Sha1Digest &root, const Sha1Digest &vs,
                  const Sha1Digest &ps, const Sha1Digest &gs) {
  Out out;
  auto d = desc;
  Sha1Digest ids[4] = {root, vs, ps, gs};
  std::vector<std::string> names;
  std::vector<D3D12_INPUT_ELEMENT_DESC> elements;
  GraphicsFields(out, d, ids, names, elements);
  return std::move(out.bytes);
}

std::vector<uint8_t>
SerializeCompute(const D3D12_COMPUTE_PIPELINE_STATE_DESC &desc, const Sha1Digest &root, const Sha1Digest &cs) {
  Out out;
  auto d = desc;
  Sha1Digest ids[2] = {root, cs};
  ComputeFields(out, d, ids);
  return std::move(out.bytes);
}

bool
ParseGraphics(const std::vector<uint8_t> &payload, GraphicsRecord &record) {
  In in{payload.data(), payload.data() + payload.size()};
  record.desc = {};
  Sha1Digest ids[4];
  GraphicsFields(in, record.desc, ids, record.names, record.elements);
  record.root = ids[0];
  record.vs = ids[1];
  record.ps = ids[2];
  record.gs = ids[3];
  return in.ok && in.p == in.end;
}

bool
ParseCompute(const std::vector<uint8_t> &payload, ComputeRecord &record) {
  In in{payload.data(), payload.data() + payload.size()};
  record.desc = {};
  Sha1Digest ids[2];
  ComputeFields(in, record.desc, ids);
  record.root = ids[0];
  record.cs = ids[1];
  return in.ok && in.p == in.end;
}

Recording
Read(const std::wstring &path) {
  Recording recording;
  HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE)
    return recording;
  std::vector<uint8_t> bytes;
  bool read = ReadAll(file, bytes);
  CloseHandle(file);
  if (!read || bytes.size() < sizeof(kMagic) || memcmp(bytes.data(), kMagic, sizeof(kMagic)))
    return recording;
  recording.opened = true;
  for (size_t at = sizeof(kMagic); at < bytes.size();) {
    if (bytes.size() - at < kHeaderSize) {
      recording.bad++;
      break;
    }
    uint32_t kind, size;
    uint64_t sum;
    Sha1Digest id;
    memcpy(&kind, bytes.data() + at, 4);
    memcpy(&size, bytes.data() + at + 4, 4);
    memcpy(&sum, bytes.data() + at + 8, 8);
    memcpy(id.data, bytes.data() + at + 16, 20);
    if (size > bytes.size() - at - kHeaderSize) { // cut short
      recording.bad++;
      break;
    }
    const uint8_t *payload = bytes.data() + at + kHeaderSize;
    at += kHeaderSize + size;
    if (kind < kBlob || kind > kCompute || Checksum(payload, size) != sum) {
      recording.bad++;
      continue;
    }
    std::vector<uint8_t> copy(payload, payload + size);
    if (kind == kBlob)
      recording.blobs.emplace(id, std::move(copy));
    else
      recording.pipelines.emplace_back(Kind(kind), std::move(copy));
  }
  return recording;
}

bool
RecordingOn() {
  return !RecordingPath().empty();
}

void
RecordGraphics(const D3D12_GRAPHICS_PIPELINE_STATE_DESC &desc, const Blob &root, const Blob &vs, const Blob &ps,
               const Blob &gs) {
  if (RecordingOn())
    Append(SerializeGraphics(desc, IdOf(root), IdOf(vs), IdOf(ps), IdOf(gs)), kGraphics, {&root, &vs, &ps, &gs});
}

void
RecordCompute(const D3D12_COMPUTE_PIPELINE_STATE_DESC &desc, const Blob &root, const Blob &cs) {
  if (RecordingOn())
    Append(SerializeCompute(desc, IdOf(root), IdOf(cs)), kCompute, {&root, &cs});
}

} // namespace dxmt::record
