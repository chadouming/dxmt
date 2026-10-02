/*
 * Copyright 2026 Feifan He for CodeWeavers
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

#include "Metal.hpp"
#include "thread.hpp"
#include <atomic>
#include <cstdint>

namespace dxmt {

class Fence {
public:
  // MacNeutron (GPU overlap spec §3.11): one generation of the fence's Metal events. The shared event is what the CPU
  // sees; queues wait on the MTLEvent, which releases another queue in under 1 us (the shared event: 130-150 us).
  // Both keep the highest value signaled, so a lower value starts a new generation.
  struct Generation {
    WMT::Reference<WMT::SharedEvent> shared;
    WMT::Reference<WMT::Event> gpu;
  };

  void incRef();
  void decRef();

  Generation
  current() const {
    std::lock_guard<dxmt::mutex> lock(mutex_);
    return gen_;
  }

  // A value asked of the fence: a queue Signal as it is encoded, or the CPU's. A value lower than the last asked
  // starts a new generation. Returns the generation the value goes to.
  Generation ask(uint64_t value);

  WMT::Reference<WMT::SharedEvent>
  sharedEvent() const {
    return current().shared;
  }

  uint64_t
  completedValue() const {
    return current().shared.signaledValue();
  }

  // The CPU's Signal: the shared event, then the MTLEvent through the helper queue.
  void
  signal(uint64_t value) {
    auto gen = ask(value);
    gen.shared.signalValue(value);
    if (value)
      forward(gen, value);
  }

  // Signals a generation's MTLEvent from the CPU: a command buffer of one signal on the helper queue.
  void forward(const Generation &gen, uint64_t value);

  void
  wait(uint64_t value, uint64_t timeout = ~0ULL) const {
    current().shared.waitUntilSignaledValue(value, timeout);
  }

  Fence(WMT::Device device, WMT::CommandQueue helper);

private:
  WMT::Device device_;
  WMT::CommandQueue helper_; // the device's; it outlives its fences
  mutable dxmt::mutex mutex_;
  Generation gen_;
  uint64_t last_ = 0;
  std::atomic<uint32_t> refcount_ = {0u};
};

class EventListener {
public:
  EventListener();
  ~EventListener();

  void setEventOnValue(Fence const *fence, HANDLE event, uint64_t value);
  // Sets the event once all (or any) of the fences reach their values.
  void setEventOnValues(Fence const *const *fences, const uint64_t *values, uint32_t count, bool all, HANDLE event);

private:
  obj_handle_t shared_event_listener_;
  dxmt::thread event_listener_thread_;
};

}; // namespace dxmt
