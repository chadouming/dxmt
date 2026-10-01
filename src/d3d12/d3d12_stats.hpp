#pragma once

#include <cstdint>

namespace dxmt {

// MacNeutron: DXMT_STATS=1 (with DXMT_DXIL_DUMP=<folder>) times every D3D12 call per thread and appends a report to
// <folder>/stats.txt every 5 seconds of presents, and once at exit: calls and microseconds per frame of each call,
// the time each thread spends in D3D12 (outermost calls only), counters such as encoder boundaries with no barrier,
// and Metal encoders labelled "pass-<n> <kind>" (n counts as F9's passes.txt does) for Instruments. Off: one branch.
extern bool g_stats_on;

unsigned StatId(const char *name); // registers a name; call sites keep the id in a static
void StatCount(unsigned id, uint64_t n = 1); // a counter: no time
void StatsFrame();         // at each present
unsigned StatsNextPass(); // numbers encoders from 0 at each present, as passes.txt does
uint64_t StatBegin();
void StatEnd(unsigned id, uint64_t t0);

struct StatScope {
  unsigned id;
  uint64_t t0;
  StatScope(unsigned id) : id(id), t0(g_stats_on ? StatBegin() : 0) {}
  ~StatScope() {
    if (g_stats_on)
      StatEnd(id, t0);
  }
};

} // namespace dxmt

#define DXMT_STAT_SCOPE(name)                                                                                          \
  static const unsigned dxmt_stat_id_ = ::dxmt::StatId(name);                                                          \
  ::dxmt::StatScope dxmt_stat_scope_(dxmt_stat_id_)
#define DXMT_STAT_COUNT(name, n)                                                                                       \
  do {                                                                                                                 \
    if (::dxmt::g_stats_on) {                                                                                          \
      static const unsigned dxmt_stat_id_ = ::dxmt::StatId(name);                                                      \
      ::dxmt::StatCount(dxmt_stat_id_, n);                                                                             \
    }                                                                                                                  \
  } while (0)
