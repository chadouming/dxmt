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

#include "d3d12_shader_cache.hpp"
#include "dxmt_shader_cache.hpp"
#include "log/log.hpp"
#include "util_env.hpp"
#include "util_string.hpp"
#include "DXBCParser/BlobContainer.h"
#include <atomic>
#include <cstring>

namespace dxmt {

namespace {

std::atomic<uint64_t> function_hits, function_misses, reflection_hits, reflection_misses, function_lookups;

// D3D12 compiles for Metal 3.1 (SM50_SHADER_METAL_310), so its entries live in that store.
ShaderCache &
Store() {
  return ShaderCache::getInstance(WMTMetal310);
}

using Key = std::pair<Sha1Digest, Sha1Digest>; // the store's key, as D3D11's

Sha1Digest
Tag(const char *tag) {
  return Sha1HashState::compute(tag, strlen(tag));
}

// A probed run (DXMT_PROBE, frame debugging) translates pixel shaders into debug colours: it neither reads nor stores
// functions, or a later normal run would draw with them.
bool
ProbeOn() {
  static const bool on = !env::getEnvVar("DXMT_PROBE").empty();
  return on;
}

void
CountFunctionLookup() {
  if (++function_lookups % 1000 == 0)
    LogShaderCacheCounters();
}

} // namespace

HRESULT
CachedShader::Initialize(const D3D12_SHADER_BYTECODE &bytecode) {
  microsoft::CDXBCParser parser;
  if (HRESULT hr = parser.ReadDXBC(bytecode.pShaderBytecode, bytecode.BytecodeLength); FAILED(hr))
    return hr;
  bytecode_ = bytecode;
  digest_ = Sha1HashState::compute(bytecode.pShaderBytecode, bytecode.BytecodeLength);
  return S_OK;
}

void
CachedShader::Keep() {
  auto bytes = static_cast<const char *>(bytecode_.pShaderBytecode);
  kept_.assign(bytes, bytes + bytecode_.BytecodeLength);
  bytecode_.pShaderBytecode = kept_.data();
}

HRESULT
CachedShader::Parse(sm50_shader_t *out) {
  if (!parsed_) {
    SM50Error error;
    if (SM50Initialize(bytecode_.pShaderBytecode, bytecode_.BytecodeLength, &shader_, &reflection_, &error)) {
      ERR("Failed to initialize shader: ", SM50GetErrorMessageString(error));
      return E_FAIL;
    }
    parsed_ = reflected_ = true;
  }
  *out = shader_;
  return S_OK;
}

HRESULT
CachedShader::Reflection(MTL_SHADER_REFLECTION *out) {
  if (!reflected_) {
    Key key{digest_, Tag("d3d12-reflection")};
    bool enabled = false, cached = false;
    if (auto reader = Store().getReader()) {
      enabled = true;
      auto data = reader->get(key);
      cached = data && data.copyBytes(&reflection_, sizeof(reflection_)) == sizeof(reflection_);
    }
    if (cached) {
      reflection_hits++;
      reflected_ = true;
    } else {
      if (enabled)
        reflection_misses++;
      sm50_shader_t shader;
      if (HRESULT hr = Parse(&shader); FAILED(hr))
        return hr;
      if (auto writer = Store().getWriter())
        writer->set(key, WMT::MakeDispatchData(&reflection_, sizeof(reflection_)));
    }
  }
  *out = reflection_;
  return S_OK;
}

std::optional<Sha1Digest>
HashCompileArgs(const SM50_SHADER_COMPILATION_ARGUMENT_DATA *args) {
  // Fields one by one, never whole structs, so padding never enters a key.
  Sha1HashState h;
  for (auto *arg = args; arg; arg = static_cast<const SM50_SHADER_COMPILATION_ARGUMENT_DATA *>(arg->next)) {
    uint32_t type = arg->type;
    h.update(type);
    switch (arg->type) {
    case SM50_SHADER_COMMON: {
      auto *d = reinterpret_cast<const SM50_SHADER_COMMON_DATA *>(arg);
      uint32_t flags = d->flags, version = d->metal_version;
      h.update(flags).update(version);
      break;
    }
    case SM50_SHADER_ROOT_SIGNATURE: {
      auto *d = reinterpret_cast<const SM50_SHADER_ROOT_SIGNATURE_DATA *>(arg);
      uint64_t length = d->bytecode_length;
      h.update(length).update(d->bytecode, d->bytecode_length);
      break;
    }
    case SM50_SHADER_IA_INPUT_LAYOUT: {
      auto *d = reinterpret_cast<const SM50_SHADER_IA_INPUT_LAYOUT_DATA *>(arg);
      uint32_t index_format = d->index_buffer_format;
      h.update(index_format).update(d->slot_mask).update(d->num_elements);
      for (uint32_t i = 0; i < d->num_elements; i++) {
        auto &e = d->elements[i];
        uint32_t step_function = e.step_function, step_rate = e.step_rate;
        h.update(e.reg).update(e.slot).update(e.aligned_byte_offset).update(e.format).update(step_function)
            .update(step_rate);
      }
      break;
    }
    case SM50_SHADER_PSO_PIXEL_SHADER: {
      auto *d = reinterpret_cast<const SM50_SHADER_PSO_PIXEL_SHADER_DATA *>(arg);
      uint8_t dual_source = d->dual_source_blending, no_depth = d->disable_depth_output;
      h.update(d->sample_mask).update(dual_source).update(no_depth).update(d->unorm_output_reg_mask);
      for (uint32_t format : d->pixel_formats)
        h.update(format);
      break;
    }
    case SM50_SHADER_PSO_GEOMETRY_SHADER: {
      uint8_t strip = reinterpret_cast<const SM50_SHADER_PSO_GEOMETRY_SHADER_DATA *>(arg)->strip_topology;
      h.update(strip);
      break;
    }
    default:
      return std::nullopt;
    }
  }
  return h.final();
}

HRESULT
CompileFunction(WMT::Device device, FunctionKind kind, CachedShader &first, CachedShader *second,
                SM50_SHADER_COMPILATION_ARGUMENT_DATA *args, const char *name, const char *stage,
                WMT::Reference<WMT::Function> &function) {
  WMT::Reference<WMT::Error> err;
  std::optional<Key> key;
  if (auto variant = HashCompileArgs(args); variant && !ProbeOn()) {
    Sha1HashState shaders, v;
    shaders.update(first.digest());
    if (second)
      shaders.update(second->digest());
    v.update(Tag("d3d12-function")).update(name, strlen(name)).update(*variant);
    // Translation experiments (GPU efficiency spec E3, E5) translate differently: they key their own entries.
    static const std::string experiments = env::getEnvVar("DXMT_DXIL_VS_FAST") + "/" + env::getEnvVar("DXMT_DXIL_BOUNDS");
    if (experiments != "/")
      v.update(experiments.data(), experiments.size());
    key = Key{shaders.final(), v.final()};
  }
  if (key) {
    // ponytail: one reader connection behind one lock; a connection per thread if the lock shows up in profiles.
    WMT::Reference<WMT::DispatchData> data;
    bool enabled = false;
    if (auto reader = Store().getReader()) {
      enabled = true;
      data = reader->get(*key);
    }
    if (data) {
      function = device.newLibrary(data, err).newFunction(name);
      if (function) {
        function_hits++;
        CountFunctionLookup();
        return S_OK;
      }
      static std::atomic_flag warned = ATOMIC_FLAG_INIT;
      if (!warned.test_and_set())
        WARN("d3d12 shader cache: rejected a cached function, recompiling");
    }
    if (enabled) {
      function_misses++;
      CountFunctionLookup();
    }
  }

  sm50_shader_t a = {}, b = {};
  HRESULT hr;
  if (FAILED(hr = first.Parse(&a)) || (second && FAILED(hr = second->Parse(&b))))
    return hr;
  SM50ShaderBitcode bitcode;
  SM50Error error;
  int failed = 1;
  switch (kind) {
  case FunctionKind::Shader:
    failed = SM50Compile(a, args, name, &bitcode, &error);
    break;
  case FunctionKind::GeometryVertex:
    failed = SM50CompileGeometryPipelineVertex(a, b, args, name, &bitcode, &error);
    break;
  case FunctionKind::GeometryMesh:
    failed = SM50CompileGeometryPipelineGeometry(a, b, args, name, &bitcode, &error);
    break;
  }
  if (failed)
    return ShaderCompileFailed(stage, error);
  SM50_COMPILED_BITCODE compiled;
  SM50GetCompiledBitcode(bitcode, &compiled);
  auto data = WMT::MakeDispatchData(compiled.Data, compiled.Size);
  function = device.newLibrary(data, err).newFunction(name);
  if (function && key)
    if (auto writer = Store().getWriter())
      writer->set(*key, data);
  return S_OK;
}

void
LogShaderCacheCounters() {
  uint64_t fh = function_hits, fm = function_misses, rh = reflection_hits, rm = reflection_misses;
  if (fh + fm + rh + rm == 0)
    return;
  Logger::info(str::format("d3d12 shader cache: functions ", fh, " hit ", fm, " missed, reflections ", rh, " hit ", rm,
                           " missed"));
}

} // namespace dxmt
