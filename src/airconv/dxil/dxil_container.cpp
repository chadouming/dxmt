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
#include "dxil.hpp"
#include "../airconv_error.hpp"
#include "DXBCParser/BlobContainer.h"
#include <cstring>

namespace dxmt::dxil {

llvm::Expected<std::optional<Container>>
FindDXIL(const void *bytecode, size_t size) {
  microsoft::CDXBCParser parser;
  if (FAILED(parser.ReadDXBC(bytecode, (UINT32)size)))
    return std::nullopt; // the DXBC path reports its own error
  UINT32 index = parser.FindNextMatchingBlob(microsoft::DXBC_DXIL);
  if (index == DXBC_BLOB_NOT_FOUND)
    return std::nullopt;
  auto part = static_cast<const char *>(parser.GetBlob(index));
  size_t part_size = parser.GetBlobSize(index);
  // DxilProgramHeader: ProgramVersion, SizeInUint32, then DxilBitcodeHeader: "DXIL", DxilVersion, BitcodeOffset, BitcodeSize.
  uint32_t header[6];
  if (part_size < sizeof(header))
    return llvm::make_error<UnsupportedFeature>("DXIL: program header truncated");
  memcpy(header, part, sizeof(header));
  if (memcmp(part + 8, "DXIL", 4))
    return llvm::make_error<UnsupportedFeature>("DXIL: bad program header");
  uint64_t start = 8 + uint64_t(header[4]), length = header[5];
  if (length < 4 || start + length > part_size)
    return llvm::make_error<UnsupportedFeature>("DXIL: bitcode lies outside the DXIL part");
  return Container{part + start, (size_t)length};
}

} // namespace dxmt::dxil
