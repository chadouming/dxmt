#include "d3d12_dxil_dump.hpp"
#include "util_env.hpp"
#include "util_string.hpp"
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

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

// DXMT_DXIL_DUMP as a Windows path, created on first use; empty when capture is off. Wide, because the variable
// arrives as UTF-8 and the ANSI APIs would misread any name outside ASCII.
const std::wstring &CaptureFolder() {
  static const std::wstring folder = [] {
    std::string value = env::getEnvVar("DXMT_DXIL_DUMP");
    // Launch options carry Mac paths; Wine's Z: drive is the Mac's root.
    if (!value.empty() && value[0] == '/')
      value = "Z:" + value;
    std::wstring wide = str::tows(value.c_str());
    if (!wide.empty())
      CreateDirectoryW(wide.c_str(), nullptr); // may exist already; any other failure shows as no captures
    return wide;
  }();
  return folder;
}

} // namespace

bool DXILCaptureMode() {
  return !CaptureFolder().empty();
}

uint64_t CaptureHash(const void *data, size_t size) {
  auto bytes = static_cast<const uint8_t *>(data);
  uint64_t hash = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < size; i++)
    hash = (hash ^ bytes[i]) * 0x100000001b3ull;
  return hash;
}

namespace {

// Saves `blob` as <folder>\<name> once (see DumpDXIL).
void SaveOnce(const char *name, const void *blob, size_t size) {
  const std::wstring &folder = CaptureFolder();
  std::wstring path = folder + L"\\" + str::tows(name);
  if (GetFileAttributesW(path.c_str()) != INVALID_FILE_ATTRIBUTES)
    return; // captured before
  // Written under a temporary name and renamed, so a capture cut short (a killed game) never takes the final name.
  // Any failure only means this shader isn't captured.
  std::wstring temp = path + L".tmp";
  HANDLE file = CreateFileW(temp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE)
    return;
  DWORD written = 0;
  bool ok = WriteFile(file, blob, (DWORD)size, &written, nullptr) && written == size;
  CloseHandle(file);
  // MoveFileW fails if another thread captured the same shader meanwhile: that capture is kept.
  if (!ok || !MoveFileW(temp.c_str(), path.c_str()))
    DeleteFileW(temp.c_str());
}

} // namespace

void DumpDXIL(const D3D12_SHADER_BYTECODE &Bytecode) {
  if (CaptureFolder().empty() || !Bytecode.pShaderBytecode)
    return;
  auto blob = static_cast<const uint8_t *>(Bytecode.pShaderBytecode);
  const char *stage = DXILStage(blob, Bytecode.BytecodeLength);
  if (!stage)
    return;
  char name[64];
  snprintf(name, sizeof(name), "%s-%016llx.dxil", stage, (unsigned long long)CaptureHash(blob, Bytecode.BytecodeLength));
  SaveOnce(name, blob, Bytecode.BytecodeLength);
}

void DumpRootSignature(const void *blob, size_t size) {
  if (CaptureFolder().empty() || !blob)
    return;
  char name[64];
  snprintf(name, sizeof(name), "rs-%016llx.bin", (unsigned long long)CaptureHash(blob, size));
  SaveOnce(name, blob, size);
}

bool MakeCaptureFolder(const char *name) {
  const std::wstring &folder = CaptureFolder();
  return folder.empty() || CreateDirectoryW((folder + L"\\" + str::tows(name)).c_str(), nullptr) ||
         GetLastError() != ERROR_ALREADY_EXISTS;
}

void SaveCapture(const char *name, const void *data, size_t size) {
  const std::wstring &folder = CaptureFolder();
  if (folder.empty())
    return;
  std::wstring path = folder + L"\\" + str::tows(name), temp = path + L".tmp";
  HANDLE file = CreateFileW(temp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE)
    return;
  DWORD written = 0;
  bool ok = WriteFile(file, data, (DWORD)size, &written, nullptr) && written == size;
  CloseHandle(file);
  if (!ok || !MoveFileExW(temp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING))
    DeleteFileW(temp.c_str());
}

namespace {
std::mutex pipeline_names_mutex;
std::unordered_map<uint64_t, std::string> pipeline_names;
} // namespace

void NamePipeline(uint64_t pso, const std::string &name) {
  if (!DXILCaptureMode() || !pso)
    return;
  std::lock_guard<std::mutex> lock(pipeline_names_mutex);
  pipeline_names[pso] = name;
}

std::string PipelineName(uint64_t pso) {
  std::lock_guard<std::mutex> lock(pipeline_names_mutex);
  auto found = pipeline_names.find(pso);
  if (found != pipeline_names.end())
    return found->second;
  char text[32];
  snprintf(text, sizeof(text), "pso-%llx", (unsigned long long)pso);
  return text;
}

void LogPipeline(const std::string &line) {
  const std::wstring &folder = CaptureFolder();
  if (folder.empty())
    return;
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  std::wstring path = folder + L"\\pipelines.txt";
  HANDLE file = CreateFileW(path.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE)
    return;
  std::string text = line + "\n";
  DWORD written = 0;
  WriteFile(file, text.data(), (DWORD)text.size(), &written, nullptr);
  CloseHandle(file);
}

} // namespace dxmt
