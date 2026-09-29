#pragma once

#include "d3d12.h"

namespace dxmt {

// MacNeutron: with DXMT_DXIL_DUMP=<folder>, saves each DXIL shader a pipeline is created from as
// <folder>/<stage>-<FNV-1a 64 of the blob, 16 hex>.dxil. DXBC shaders are ignored, existing files are kept, and
// write errors are ignored, so capturing never changes what the game gets.
void DumpDXIL(const D3D12_SHADER_BYTECODE &Bytecode);

// True when DXMT_DXIL_DUMP is set. The device then reports what Unreal Engine's SM6 check needs (feature level 12_1,
// shader model up to 6.7, binding tier 3, wave ops, 64-bit atomics), so SM6-only games get as far as creating their
// DXIL pipelines, which are captured and fail with E_NOTIMPL. For capture runs only: such a game can't render.
bool DXILCaptureMode();

} // namespace dxmt
