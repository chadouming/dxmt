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
// D3D12 option structs newer than llvm-mingw's d3d12.h, as the Agility SDK lays them out.
#include "d3d12.h"

namespace dxmt {

constexpr D3D12_FEATURE kFeatureOptions19 = (D3D12_FEATURE)48;
constexpr D3D12_FEATURE kFeatureOptions21 = (D3D12_FEATURE)53;

struct D3D12_FEATURE_DATA_D3D12_OPTIONS19_MN {
  BOOL MismatchingOutputDimensionsSupported;
  UINT SupportedSampleCountsWithNoOutputs;
  BOOL PointSamplingAddressesNeverRoundUp;
  BOOL RasterizerDesc2Supported;
  BOOL NarrowQuadrilateralLinesSupported;
  BOOL AnisoFilterWithPointMipSupported;
  UINT MaxSamplerDescriptorHeapSize;
  UINT MaxSamplerDescriptorHeapSizeWithStaticSamplers;
  UINT MaxViewDescriptorHeapSize;
  BOOL ComputeOnlyCustomHeapSupported;
};
static_assert(sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS19_MN) == 40);

struct D3D12_FEATURE_DATA_D3D12_OPTIONS21_MN {
  UINT WorkGraphsTier;      // D3D12_WORK_GRAPHS_TIER: 0, not supported
  UINT ExecuteIndirectTier; // D3D12_EXECUTE_INDIRECT_TIER: 10, tier 1.0
  BOOL SampleCmpGradientAndBiasSupported;
  BOOL ExtendedCommandInfoSupported;
};
static_assert(sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS21_MN) == 16);

} // namespace dxmt
