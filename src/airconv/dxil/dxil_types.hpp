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
#include <cctype>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

// DXIL (Shader Model 6) front end for airconv. Enum values are DXC's (DxilConstants.h).
namespace dxmt::dxil {

enum class ShaderKind : uint32_t { Pixel = 0, Vertex = 1, Geometry = 2, Hull = 3, Domain = 4, Compute = 5, Library = 6, Invalid = ~0u };
enum class ResourceClass : uint32_t { SRV = 0, UAV = 1, CBuffer = 2, Sampler = 3 };
enum class ResourceKind : uint32_t {
  Invalid = 0, Texture1D, Texture2D, Texture2DMS, Texture3D, TextureCube, Texture1DArray, Texture2DArray, Texture2DMSArray,
  TextureCubeArray, TypedBuffer, RawBuffer, StructuredBuffer, CBuffer, Sampler, TBuffer, RTAccelerationStructure
};
enum class ComponentType : uint32_t {
  Invalid = 0, I1, I16, U16, I32, U32, I64, U64, F16, F32, F64, SNormF16, UNormF16, SNormF32, UNormF32, SNormF64, UNormF64
};
enum class SemanticKind : uint32_t {
  Arbitrary = 0, VertexID, InstanceID, Position, RenderTargetArrayIndex, ViewPortArrayIndex, ClipDistance, CullDistance,
  OutputControlPointID, DomainLocation, PrimitiveID, GSInstanceID, SampleIndex, IsFrontFace, Coverage, InnerCoverage,
  Target, Depth, DepthLessEqual, DepthGreaterEqual, StencilRef, DispatchThreadID, GroupID, GroupIndex, GroupThreadID
};
enum class Interpolation : uint32_t {
  Undefined = 0, Constant, Linear, LinearCentroid, LinearNoperspective, LinearNoperspectiveCentroid, LinearSample,
  LinearNoperspectiveSample
};

namespace op { // dx.op opcodes: the first argument of every dx.op call
enum : uint32_t {
  LoadInput = 4, StoreOutput = 5, FAbs = 6, Saturate = 7, IsNaN = 8, IsInf = 9, IsFinite = 10, IsNormal = 11,
  Cos = 12, Sin = 13, Tan = 14, Acos = 15, Asin = 16, Atan = 17, Hcos = 18, Hsin = 19, Htan = 20, Exp = 21, Frc = 22,
  Log = 23, Sqrt = 24, Rsqrt = 25, Round_ne = 26, Round_ni = 27, Round_pi = 28, Round_z = 29, Bfrev = 30,
  Countbits = 31, FirstbitLo = 32, FirstbitHi = 33, FirstbitSHi = 34, FMax = 35, FMin = 36, IMax = 37, IMin = 38,
  UMax = 39, UMin = 40, IMul = 41, UMul = 42, UDiv = 43, UAddc = 44, USubb = 45, FMad = 46, Fma = 47, IMad = 48,
  UMad = 49, Msad = 50, Ibfe = 51, Ubfe = 52, Bfi = 53, Dot2 = 54, Dot3 = 55, Dot4 = 56, CreateHandle = 57,
  CBufferLoad = 58, CBufferLoadLegacy = 59, Sample = 60, SampleBias = 61, SampleLevel = 62, SampleGrad = 63,
  SampleCmp = 64, SampleCmpLevelZero = 65, TextureLoad = 66, TextureStore = 67, BufferLoad = 68, BufferStore = 69,
  BufferUpdateCounter = 70, CheckAccessFullyMapped = 71, GetDimensions = 72, TextureGather = 73, TextureGatherCmp = 74,
  AtomicBinOp = 78, AtomicCompareExchange = 79, Barrier = 80, CalculateLOD = 81, Discard = 82, DerivCoarseX = 83,
  DerivCoarseY = 84, DerivFineX = 85, DerivFineY = 86, SampleIndex = 90, Coverage = 91, ThreadId = 93, GroupId = 94,
  ThreadIdInGroup = 95, FlattenedThreadIdInGroup = 96, MakeDouble = 101, SplitDouble = 102, PrimitiveID = 108,
  WaveIsFirstLane = 110, WaveGetLaneIndex = 111, WaveGetLaneCount = 112, WaveAnyTrue = 113, WaveAllTrue = 114,
  WaveActiveAllEqual = 115, WaveActiveBallot = 116, WaveReadLaneAt = 117, WaveReadLaneFirst = 118, WaveActiveOp = 119,
  WaveActiveBit = 120, WavePrefixOp = 121, BitcastI16toF16 = 124, BitcastF16toI16 = 125, BitcastI32toF32 = 126,
  BitcastF32toI32 = 127, LegacyF32ToF16 = 130, LegacyF16ToF32 = 131, WaveAllBitCount = 135, WavePrefixBitCount = 136,
  RawBufferLoad = 139, RawBufferStore = 140, Dot2AddHalf = 162, Dot4AddI8Packed = 163, Dot4AddU8Packed = 164,
  AnnotateHandle = 216, CreateHandleFromBinding = 217, CreateHandleFromHeap = 218, Unpack4x8 = 219, Pack4x8 = 220,
  IsHelperLane = 221
};
} // namespace op

struct SignatureElement {
  uint32_t id = 0;
  std::string name;                       // semantic name, e.g. "TEXCOORD"
  std::vector<uint32_t> semantic_indices; // one per row
  ComponentType type = ComponentType::Invalid;
  SemanticKind kind = SemanticKind::Arbitrary;
  Interpolation interpolation = Interpolation::Undefined;
  uint32_t rows = 0, cols = 0;
  int32_t start_row = -1; // -1: a system value that isn't packed
  int32_t start_col = -1;
};

struct Resource {
  uint32_t id = 0; // the range ID: createHandle's rangeId, and airconv's RangeId
  ResourceClass cls = ResourceClass::SRV;
  ResourceKind kind = ResourceKind::Invalid;
  uint32_t space = 0, lower_bound = 0, size = 1; // size ~0u = unbounded
  ComponentType element_type = ComponentType::Invalid; // typed textures and buffers
  uint32_t stride = 0;                                  // structured buffers
  uint32_t cbuffer_size = 0;                            // constant buffers, in bytes
  bool globally_coherent = false, has_counter = false, rasterizer_ordered = false, comparison_sampler = false;
};

struct EntryInfo {
  ShaderKind kind = ShaderKind::Invalid;
  uint32_t sm_major = 0, sm_minor = 0;
  std::string name;
  uint32_t numthreads[3] = {0, 0, 0};
  std::vector<SignatureElement> inputs, outputs;
  std::vector<Resource> resources; // all four classes

  const Resource *FindResource(ResourceClass cls, uint32_t id) const {
    for (auto &r : resources)
      if (r.cls == cls && r.id == id)
        return &r;
    return nullptr;
  }
};

struct DXILShader {
  std::vector<char> bitcode; // parsed again by every SM50Compile, which has its own context
  EntryInfo entry;
  // Input elements that aren't in a signature register (unpacked system values): element ID -> function argument,
  // or one of the kUnpacked* values below.
  std::map<uint32_t, uint32_t> unpacked_inputs;
  // Function arguments for the simdgroup lane index and size, when the shader uses wave operations; ~0u otherwise.
  uint32_t lane_index_arg = ~0u, lane_count_arg = ~0u;
};

// Vertex shaders' unpacked SV_VertexID/SV_InstanceID read io_binding_map's vertex_id/instance_id instead of an argument.
constexpr uint32_t kUnpackedVertexID = ~0u, kUnpackedInstanceID = ~0u - 1;

// Varyings link by semantic, not register (DXIL packs each stage's signature on its own): "TEXCOORD0".
inline std::string UserName(const SignatureElement &e, uint32_t row) {
  std::string name = e.name;
  for (auto &c : name)
    c = (char)toupper((unsigned char)c);
  return name + std::to_string(row < e.semantic_indices.size() ? e.semantic_indices[row] : row);
}

} // namespace dxmt::dxil
