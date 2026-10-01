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
#include "Metal.hpp"
#include "airconv_public.h"
#include "d3d12_pipeline.hpp"
#include "sha1/sha1_util.hpp"
#include <optional>
#include <vector>

namespace dxmt {

// One shader stage's bytecode during pipeline creation (shader pre-caching spec §3.3): its SHA-1, its reflection from
// the translation cache or airconv, and the airconv shader, parsed only when a function misses.
class CachedShader {
public:
  // Checks the container and hashes it; fails as the container check always has.
  HRESULT Initialize(const D3D12_SHADER_BYTECODE &bytecode);
  // Copies the bytecode, for a shader compiled after creation returns (the geometry pipeline's variants).
  void Keep();
  HRESULT Reflection(MTL_SHADER_REFLECTION *out);
  // The airconv shader, parsed on first use.
  HRESULT Parse(sm50_shader_t *out);
  const Sha1Digest &digest() const { return digest_; }

private:
  D3D12_SHADER_BYTECODE bytecode_ = {};
  std::vector<char> kept_;
  Sha1Digest digest_ = {};
  SM50Shader shader_;
  bool parsed_ = false;
  bool reflected_ = false;
  MTL_SHADER_REFLECTION reflection_ = {};
};

// The variant digest of an airconv argument chain (spec §3.2); nullopt for an argument type it doesn't know.
std::optional<Sha1Digest> HashCompileArgs(const SM50_SHADER_COMPILATION_ARGUMENT_DATA *args);

enum class FunctionKind { Shader, GeometryVertex, GeometryMesh };

// A translated function, from the translation cache or compiled by airconv and stored (spec §3.3). `second` is the
// geometry shader for the geometry pipeline's two compiles, else null; `stage` names the shader in error messages.
HRESULT CompileFunction(WMT::Device device, FunctionKind kind, CachedShader &first, CachedShader *second,
                        SM50_SHADER_COMPILATION_ARGUMENT_DATA *args, const char *name, const char *stage,
                        WMT::Reference<WMT::Function> &function);

// "d3d12 shader cache: functions <h> hit <m> missed, reflections <h> hit <m> missed" (spec §3.4); nothing when this
// process made no lookup.
void LogShaderCacheCounters();

} // namespace dxmt
