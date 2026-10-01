#include "d3d12_stats.hpp"
#include "d3d12_dxil_dump.hpp"
#include "util_env.hpp"
#include "util_string.hpp"
#include <windows.h>
#include <x86intrin.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace dxmt {

bool g_stats_on = env::getEnvVar("DXMT_STATS") == "1";

namespace {

constexpr unsigned kMaxStats = 1024; // names past this share the last slot

// One per thread that calls into D3D12; written only by its thread, read racily by the reporter (stats only).
struct ThreadStats {
  DWORD tid;
  unsigned depth = 0;
  uint64_t outer = 0, prev_outer = 0; // time in outermost calls
  uint64_t calls[kMaxStats] = {}, ticks[kMaxStats] = {}, max[kMaxStats] = {};
  uint64_t prev_calls[kMaxStats] = {}, prev_ticks[kMaxStats] = {};
};

std::mutex g_mutex;
std::vector<const char *> g_names;
std::vector<ThreadStats *> g_threads;
thread_local ThreadStats *t_stats = nullptr;
uint64_t g_frames = 0, g_prev_frames = 0, g_last_tsc = 0;
LARGE_INTEGER g_last_qpc = {};
std::string g_text;

ThreadStats &
Mine() {
  if (!t_stats) {
    t_stats = new ThreadStats();
    t_stats->tid = GetCurrentThreadId();
    std::lock_guard<std::mutex> lock(g_mutex);
    g_threads.push_back(t_stats);
  }
  return *t_stats;
}

std::string
ThreadName(DWORD tid) {
  std::string name;
  if (HANDLE thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, tid)) {
    PWSTR description = nullptr;
    if (SUCCEEDED(GetThreadDescription(thread, &description)) && description) {
      name = str::fromws(description);
      LocalFree(description);
    }
    CloseHandle(thread);
  }
  return name.empty() ? "thread" : name;
}

void
Report(const char *why) {
  LARGE_INTEGER qpc, freq;
  QueryPerformanceCounter(&qpc);
  QueryPerformanceFrequency(&freq);
  uint64_t tsc = __rdtsc();
  double seconds = double(qpc.QuadPart - g_last_qpc.QuadPart) / freq.QuadPart;
  double per_us = seconds > 0 ? (tsc - g_last_tsc) / (seconds * 1e6) : 1;
  uint64_t frames = g_frames - g_prev_frames;
  std::string out;
  char line[512];
  SYSTEMTIME now;
  GetLocalTime(&now);
  snprintf(line, sizeof(line), "# %02u:%02u:%02u %.1f s, %llu frames (%.1f fps), %s\n", now.wHour, now.wMinute,
           now.wSecond, seconds, (unsigned long long)frames, seconds > 0 ? frames / seconds : 0.0, why);
  out += line;
  std::lock_guard<std::mutex> lock(g_mutex);
  std::vector<uint64_t> counters(g_names.size());
  for (auto *t : g_threads) {
    uint64_t outer = t->outer - t->prev_outer;
    std::vector<std::pair<uint64_t, unsigned>> timed;
    for (unsigned id = 0; id < g_names.size() && id < kMaxStats; id++) {
      uint64_t calls = t->calls[id] - t->prev_calls[id];
      if (!calls)
        continue;
      if (g_names[id][0] == '#')
        counters[id] += calls;
      else
        timed.push_back({t->ticks[id] - t->prev_ticks[id], id});
    }
    if (!timed.empty()) {
      snprintf(line, sizeof(line), "thread %s (%lu): %.0f us in D3D12", ThreadName(t->tid).c_str(),
               (unsigned long)t->tid, outer / per_us);
      out += line;
      if (frames) {
        snprintf(line, sizeof(line), ", %.1f us/frame", outer / per_us / frames);
        out += line;
      }
      out += "\n";
      std::sort(timed.begin(), timed.end(), std::greater<>());
      for (size_t i = 0; i < timed.size() && i < 30; i++) {
        unsigned id = timed[i].second;
        uint64_t calls = t->calls[id] - t->prev_calls[id];
        snprintf(line, sizeof(line), "  %s calls %llu us %.1f max-us %.1f", g_names[id], (unsigned long long)calls,
                 timed[i].first / per_us, t->max[id] / per_us);
        out += line;
        if (frames) {
          snprintf(line, sizeof(line), " | per frame: calls %.1f us %.1f", double(calls) / frames,
                   timed[i].first / per_us / frames);
          out += line;
        }
        out += "\n";
      }
    }
    t->prev_outer = t->outer;
    for (unsigned id = 0; id < g_names.size() && id < kMaxStats; id++) {
      t->prev_calls[id] = t->calls[id];
      t->prev_ticks[id] = t->ticks[id];
      t->max[id] = 0;
    }
  }
  // Counters by name (several call sites may count the same thing).
  std::map<std::string, uint64_t> totals;
  for (unsigned id = 0; id < counters.size(); id++)
    if (counters[id])
      totals[g_names[id] + 1] += counters[id];
  out += "counters:\n";
  for (auto &[name, n] : totals) {
    snprintf(line, sizeof(line), "  %s %llu", name.c_str(), (unsigned long long)n);
    out += line;
    if (frames) {
      snprintf(line, sizeof(line), " (%.1f per frame)", double(n) / frames);
      out += line;
    }
    out += "\n";
  }
  snprintf(line, sizeof(line), "  encoder boundaries %llu, %llu with no barrier\n",
           (unsigned long long)totals["encoder boundaries"],
           (unsigned long long)totals["encoder boundaries with no barrier"]);
  out += line;
  g_text += out;
  SaveCapture("stats.txt", g_text.data(), g_text.size());
  g_prev_frames = g_frames;
  g_last_tsc = tsc;
  g_last_qpc = qpc;
}

// Starts the clock at load, and reports once more at exit (the capture folder is set up first so it outlives this).
struct Lifetime {
  Lifetime() {
    if (!g_stats_on)
      return;
    DXILCaptureMode();
    QueryPerformanceCounter(&g_last_qpc);
    g_last_tsc = __rdtsc();
  }
  ~Lifetime() {
    if (g_stats_on)
      Report("exit");
  }
} g_lifetime;

} // namespace

unsigned
StatId(const char *name) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_names.push_back(name);
  return std::min<unsigned>(g_names.size() - 1, kMaxStats - 1);
}

void
StatCount(unsigned id, uint64_t n) {
  Mine().calls[id] += n;
}

uint64_t
StatBegin() {
  Mine().depth++;
  return __rdtsc();
}

void
StatEnd(unsigned id, uint64_t t0) {
  uint64_t d = __rdtsc() - t0;
  auto &m = Mine();
  m.calls[id]++;
  m.ticks[id] += d;
  if (d > m.max[id])
    m.max[id] = d;
  if (--m.depth == 0)
    m.outer += d;
}

std::atomic_uint g_pass = 0;

unsigned
StatsNextPass() {
  return g_pass++;
}

void
StatsFrame() {
  if (!g_stats_on)
    return;
  g_pass = 0;
  g_frames++;
  LARGE_INTEGER qpc, freq;
  QueryPerformanceCounter(&qpc);
  QueryPerformanceFrequency(&freq);
  if (qpc.QuadPart - g_last_qpc.QuadPart >= 5 * freq.QuadPart)
    Report("every 5 s");
}

} // namespace dxmt
