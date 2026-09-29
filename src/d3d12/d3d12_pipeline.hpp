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
#include "airconv_public.h"
#include "d3d12_device.hpp"
#include "d3d12_dxil_dump.hpp"
#include <cstdio>
#include <string>
#include "log/log.hpp"

namespace dxmt {

class SM50Shader {
  sm50_shader_t sm50_shader_{};

public:
  sm50_shader_t *
  operator&() {
    return &sm50_shader_;
  }

  operator sm50_shader_t() {
    return sm50_shader_;
  }

  ~SM50Shader() {
    if (sm50_shader_)
      SM50Destroy(sm50_shader_);
  }
};

class SM50ShaderBitcode {
  sm50_bitcode_t sm50_bitcode_{};

public:
  sm50_bitcode_t *
  operator&() {
    return &sm50_bitcode_;
  }

  operator sm50_bitcode_t() {
    return sm50_bitcode_;
  }

  ~SM50ShaderBitcode() {
    if (sm50_bitcode_)
      SM50DestroyBitcode(sm50_bitcode_);
  }
};

class SM50Error {
  sm50_error_t sm50_error_{};

public:
  sm50_error_t *
  operator&() {
    return &sm50_error_;
  }

  operator sm50_error_t() {
    return sm50_error_;
  }

  ~SM50Error() {
    if (sm50_error_)
      SM50FreeError(sm50_error_);
  }
};

// Capture mode: "<name>=<hash>" of a shader, "" without one.
inline std::string CapturedShader(const char *name, const D3D12_SHADER_BYTECODE &b) {
  if (!b.pShaderBytecode)
    return "";
  char text[48];
  snprintf(text, sizeof(text), " %s=%016llx", name, (unsigned long long)CaptureHash(b.pShaderBytecode, b.BytecodeLength));
  return text;
}

inline std::string CapturedRootSignature(ID3D12RootSignature *rs) {
  if (!rs)
    return " rs=embedded";
  const void *blob;
  size_t size = static_cast<MTLD3D12RootSignature *>(rs)->GetBlob(&blob);
  char text[48];
  snprintf(text, sizeof(text), " rs=%016llx", (unsigned long long)CaptureHash(blob, size));
  return text;
}

// airconv failed to compile a shader: log its message. A DXIL shader DXMT can't translate yet (the DXIL front end's
// messages start with "DXIL:") gets E_NOTIMPL, anything else E_FAIL. (MacNeutron)
inline HRESULT
ShaderCompileFailed(const char *stage, SM50Error &error) {
  auto message = SM50GetErrorMessageString(error);
  ERR("Failed to compile ", stage, " shader: ", message);
  return message.rfind("DXIL:", 0) == 0 ? E_NOTIMPL : E_FAIL;
}

} // namespace dxmt