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

// dxmt-replay.exe <recording>: rebuilds every pipeline a game recorded (d3d12_pipeline_record) through our d3d12.dll,
// on every core, into that game's translation and Metal caches (shader pre-caching spec §3.6). MacNeutron's launcher
// runs it before a game after a DXMT or macOS update, as Steam runs fossilize_replay for Vulkan games.
#include "d3d12_pipeline_record.hpp"
#include "log/log.hpp"
#include <windows.h>
#include <d3d12.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <thread>
#include <unordered_map>
#include <vector>

namespace dxmt {
Logger Logger::s_instance("dxmt-replay.log");
} // namespace dxmt

using namespace dxmt;

namespace {

const Sha1Digest kNone = {};

struct Job {
  record::Kind kind;
  record::GraphicsRecord graphics;
  record::ComputeRecord compute;
  ID3D12RootSignature *root = nullptr;
  bool root_failed = false;
};

} // namespace

int
wmain(int argc, wchar_t **argv) {
  if (argc != 2) {
    printf("usage: dxmt-replay.exe <recording>\n");
    return 1;
  }
  // A recording is named after the game's executable: DXMT then resolves that game's cache folders.
  std::wstring path = argv[1], name = path.substr(path.find_last_of(L"\\/") + 1);
  const std::wstring extension = L".pipelines";
  if (name.size() <= extension.size() || name.compare(name.size() - extension.size(), extension.size(), extension)) {
    printf("replay: not a recording\n");
    return 1;
  }
  name.resize(name.size() - extension.size());
  SetEnvironmentVariableW(L"DXMT_CACHE_EXE", name.c_str());
  SetEnvironmentVariableW(L"DXMT_PIPELINE_RECORD", nullptr);

  ULONGLONG start = GetTickCount64();
  record::Recording recording = record::Read(path);
  if (!recording.opened) {
    printf("replay: not a recording\n");
    return 1;
  }
  // DXMT loads only now, after DXMT_CACHE_EXE is set: dxgi.dll picks the Metal cache folder as it loads.
  using CreateDevice = HRESULT(WINAPI *)(IUnknown *, D3D_FEATURE_LEVEL, REFIID, void **);
  HMODULE d3d12 = LoadLibraryW(L"d3d12.dll");
  auto create = d3d12 ? reinterpret_cast<CreateDevice>(GetProcAddress(d3d12, "D3D12CreateDevice")) : nullptr;
  ID3D12Device *device = nullptr;
  if (!create || FAILED(create(nullptr, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), (void **)&device))) {
    printf("replay: no D3D12 device\n");
    return 1;
  }

  // Parse every pipeline and create its root signature, one thread; a pipeline missing a blob counts as bad.
  uint64_t bad = recording.bad;
  size_t graphics = 0, compute = 0;
  std::unordered_map<Sha1Digest, ID3D12RootSignature *> roots;
  auto has = [&](const Sha1Digest &id) { return id == kNone || recording.blobs.count(id); };
  auto code = [&](const Sha1Digest &id) -> D3D12_SHADER_BYTECODE {
    if (id == kNone)
      return {};
    auto &bytes = recording.blobs.at(id);
    return {bytes.data(), bytes.size()};
  };
  std::vector<Job> jobs;
  for (auto &[kind, payload] : recording.pipelines) {
    Job job = {kind};
    bool ok;
    Sha1Digest root_id;
    if (kind == record::kGraphics) {
      auto &g = job.graphics;
      ok = record::ParseGraphics(payload, g) && has(g.root) && has(g.vs) && has(g.ps) && has(g.gs);
      if (ok) {
        g.desc.VS = code(g.vs);
        g.desc.PS = code(g.ps);
        g.desc.GS = code(g.gs);
      }
      root_id = g.root;
    } else {
      auto &c = job.compute;
      ok = record::ParseCompute(payload, c) && has(c.root) && has(c.cs);
      if (ok)
        c.desc.CS = code(c.cs);
      root_id = c.root;
    }
    if (!ok) {
      bad++;
      continue;
    }
    if (root_id != kNone) {
      auto [slot, created] = roots.try_emplace(root_id, nullptr);
      if (created) {
        auto blob = code(root_id);
        if (FAILED(device->CreateRootSignature(0, blob.pShaderBytecode, blob.BytecodeLength,
                                               __uuidof(ID3D12RootSignature), (void **)&slot->second)))
          slot->second = nullptr;
      }
      job.root = slot->second;
      job.root_failed = !job.root;
    }
    (kind == record::kGraphics ? graphics : compute)++;
    jobs.push_back(std::move(job));
  }

  std::atomic<size_t> next{0}, done{0}, created{0}, failed{0};
  auto run = [&](Job &job) {
    HRESULT hr = E_FAIL;
    ID3D12PipelineState *pso = nullptr;
    if (job.root_failed) {
      // counted below as failed
    } else if (job.kind == record::kGraphics) {
      auto &g = job.graphics;
      for (size_t i = 0; i < g.elements.size(); i++)
        g.elements[i].SemanticName = g.names[i].c_str();
      g.desc.InputLayout = {g.elements.data(), (UINT)g.elements.size()};
      g.desc.pRootSignature = job.root;
      hr = device->CreateGraphicsPipelineState(&g.desc, __uuidof(ID3D12PipelineState), (void **)&pso);
    } else {
      job.compute.desc.pRootSignature = job.root;
      hr = device->CreateComputePipelineState(&job.compute.desc, __uuidof(ID3D12PipelineState), (void **)&pso);
    }
    if (SUCCEEDED(hr)) {
      pso->Release();
      created++;
    } else {
      failed++;
    }
  };
  auto work = [&] {
    for (size_t i; (i = next++) < jobs.size();) {
      run(jobs[i]);
      size_t n = ++done;
      if (n * 10 / jobs.size() != (n - 1) * 10 / jobs.size()) {
        printf("replay progress %zu/%zu\n", n, jobs.size());
        fflush(stdout);
      }
    }
  };
  std::vector<std::thread> threads(std::max(1u, std::thread::hardware_concurrency()));
  for (auto &thread : threads)
    thread = std::thread(work);
  for (auto &thread : threads)
    thread.join();
  printf("replay: %zu pipelines (%zu graphics, %zu compute), %zu created, %zu failed, %llu bad records, %llu ms\n",
         jobs.size(), graphics, compute, created.load(), failed.load(), (unsigned long long)bad,
         (unsigned long long)(GetTickCount64() - start));
  fflush(stdout);
  return 0;
}
