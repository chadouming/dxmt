#include "d3d12_dxil_dump.hpp"
#include "util_env.hpp"
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace dxmt {

namespace {

uint32_t read32(const uint8_t *p) {
  uint32_t value;
  memcpy(&value, p, 4);
  return value;
}

// The stage in the DXIL part's program header, or nullptr for DXBC or a malformed container.
const char *DXILStage(const uint8_t *blob, size_t size) {
  static const char *const stages[] = {"ps", "vs", "gs", "hs", "ds", "cs", "lib", "raygen",
                                       "intersection", "anyhit", "closesthit", "miss", "callable", "ms", "as", "node"};
  if (size < 32 || memcmp(blob, "DXBC", 4))
    return nullptr;
  uint32_t parts = read32(blob + 28);
  for (uint64_t i = 0; i < parts && 32 + 4 * (i + 1) <= size; i++) {
    uint64_t offset = read32(blob + 32 + 4 * i);
    if (offset + 12 > size || memcmp(blob + offset, "DXIL", 4))
      continue;
    uint32_t kind = read32(blob + offset + 8) >> 16;
    return kind < sizeof(stages) / sizeof(*stages) ? stages[kind] : "unknown";
  }
  return nullptr;
}

} // namespace

void DumpDXIL(const D3D12_SHADER_BYTECODE &Bytecode) {
  static const std::string folder = [] {
    std::string value = env::getEnvVar("DXMT_DXIL_DUMP");
    // Launch options carry Mac paths; Wine's Z: drive is the Mac's root.
    return !value.empty() && value[0] == '/' ? "Z:" + value : value;
  }();
  if (folder.empty() || !Bytecode.pShaderBytecode)
    return;
  auto blob = static_cast<const uint8_t *>(Bytecode.pShaderBytecode);
  const char *stage = DXILStage(blob, Bytecode.BytecodeLength);
  if (!stage)
    return;
  uint64_t hash = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < Bytecode.BytecodeLength; i++)
    hash = (hash ^ blob[i]) * 0x100000001b3ull;
  char name[64];
  snprintf(name, sizeof(name), "\\%s-%016llx.dxil", stage, (unsigned long long)hash);
  std::string path = folder + name;
  // CREATE_NEW keeps a capture made before; any failure only means this shader isn't captured.
  HANDLE file = CreateFileA(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE)
    return;
  DWORD written = 0;
  bool ok = WriteFile(file, blob, (DWORD)Bytecode.BytecodeLength, &written, nullptr) &&
            written == Bytecode.BytecodeLength;
  CloseHandle(file);
  if (!ok)
    DeleteFileA(path.c_str());
}

} // namespace dxmt
