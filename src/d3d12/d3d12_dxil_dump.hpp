#pragma once

#include "d3d12.h"

namespace dxmt {

// MacNeutron: with DXMT_DXIL_DUMP=<folder>, saves each DXIL shader a pipeline is created from as
// <folder>/<stage>-<FNV-1a 64 of the blob, 16 hex>.dxil. DXBC shaders are ignored, existing files are kept, and
// write errors are ignored, so capturing never changes what the game gets.
void DumpDXIL(const D3D12_SHADER_BYTECODE &Bytecode);

} // namespace dxmt
