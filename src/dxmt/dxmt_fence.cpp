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

#include "dxmt_fence.hpp"
#include <thread>
#include <vector>

namespace dxmt {

void
Fence::incRef() {
  refcount_.fetch_add(1u, std::memory_order_acquire);
};

void
Fence::decRef() {
  if (refcount_.fetch_sub(1u, std::memory_order_release) == 1u)
    delete this;
};

Fence::Fence(WMT::Device device, WMT::CommandQueue helper) : device_(device), helper_(helper) {
  gen_ = {device_.newSharedEvent(), device_.newEvent()};
}

Fence::Generation
Fence::ask(uint64_t value) {
  std::lock_guard<dxmt::mutex> lock(mutex_);
  if (value < last_)
    gen_ = {device_.newSharedEvent(), device_.newEvent()};
  last_ = value;
  return gen_;
}

void
Fence::forward(const Generation &gen, uint64_t value) {
  auto pool = WMT::MakeAutoreleasePool();
  auto cmdbuf = helper_.commandBuffer();
  cmdbuf.encodeSignalEvent(gen.gpu, value);
  cmdbuf.commit();
}

EventListener::EventListener() :
    shared_event_listener_(SharedEventListener_create()),
    event_listener_thread_([this]() { SharedEventListener_start(this->shared_event_listener_); }) {};

EventListener::~EventListener() {
  // FIXME: potential deadlock if a device is created and immediately destroyed
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  SharedEventListener_destroy(shared_event_listener_);
  event_listener_thread_.join();
}

void
EventListener::setEventOnValue(Fence const *fence, HANDLE event, uint64_t value) {
  auto shared = fence->sharedEvent();
  MTLSharedEvent_setWin32EventAtValue(shared.handle, shared_event_listener_, event, value);
}

void
EventListener::setEventOnValues(Fence const *const *fences, const uint64_t *values, uint32_t count, bool all, HANDLE event) {
  std::vector<WMT::Reference<WMT::SharedEvent>> held(count); // the current generations, alive through the call
  std::vector<obj_handle_t> events(count);
  for (uint32_t i = 0; i < count; i++) {
    held[i] = fences[i]->sharedEvent();
    events[i] = held[i].handle;
  }
  MTLSharedEvent_setWin32EventAtValues(shared_event_listener_, events.data(), values, count, all ? count : 1, event);
}

}; // namespace dxmt