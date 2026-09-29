#pragma once
#include "../airconv_public.h"
// Compiles a vertex function whose outputs are exactly `PixelShader`'s inputs (same names and types, zeros, plus a
// position), so a pixel shader can be put in a Metal pipeline on its own (dxil-translate). Test use only.
#ifdef __cplusplus
extern "C" {
#endif
AIRCONV_API int DXILCompilePassThroughVertex(sm50_shader_t PixelShader, sm50_bitcode_t *ppBitcode, sm50_error_t *ppError);
#ifdef __cplusplus
}
#endif
