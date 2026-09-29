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
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/MemoryBuffer.h"

namespace dxmt::dxil {

namespace {

llvm::Error Missing(const char *what) {
  return llvm::make_error<UnsupportedFeature>(std::string("DXIL: ") + what + " missing");
}

// Integer operand of any width (DXIL mixes i1, i8 and i32 in its metadata).
int64_t Int(const llvm::MDOperand &op) {
  return llvm::mdconst::extract<llvm::ConstantInt>(op)->getSExtValue();
}

const llvm::MDNode *Node(const llvm::MDOperand &op) {
  return llvm::dyn_cast_or_null<llvm::MDNode>(op.get());
}

std::string String(const llvm::MDOperand &op) {
  auto s = llvm::dyn_cast_or_null<llvm::MDString>(op.get());
  return s ? s->getString().str() : std::string();
}

// !{i32 ID, !"Name", i8 CompType, i8 SemanticKind, !{i32 SemIdx...}, i8 Interp, i32 Rows, i8 Cols, i32 StartRow, i8 StartCol, ...}
std::vector<SignatureElement> ReadSignature(const llvm::MDNode *list) {
  std::vector<SignatureElement> elements;
  if (!list)
    return elements;
  for (auto &op : list->operands()) {
    auto e = Node(op);
    if (!e || e->getNumOperands() < 10)
      continue;
    SignatureElement s;
    s.id = Int(e->getOperand(0));
    s.name = String(e->getOperand(1));
    s.type = ComponentType(Int(e->getOperand(2)));
    s.kind = SemanticKind(Int(e->getOperand(3)));
    if (auto indices = Node(e->getOperand(4)))
      for (auto &i : indices->operands())
        s.semantic_indices.push_back(Int(i));
    s.interpolation = Interpolation(Int(e->getOperand(5)));
    s.rows = Int(e->getOperand(6));
    s.cols = Int(e->getOperand(7));
    s.start_row = Int(e->getOperand(8));
    s.start_col = Int(e->getOperand(9));
    if (s.semantic_indices.size() < s.rows)
      s.semantic_indices.resize(s.rows, 0);
    elements.push_back(std::move(s));
  }
  return elements;
}

// Tag/value pairs after a resource's fixed fields: tag 0 = element type, tag 1 = structured stride.
void ReadResourceExtra(const llvm::MDNode *extra, Resource &r) {
  if (!extra)
    return;
  for (unsigned i = 0; i + 1 < extra->getNumOperands(); i += 2) {
    switch (Int(extra->getOperand(i))) {
    case 0: r.element_type = ComponentType(Int(extra->getOperand(i + 1))); break;
    case 1: r.stride = Int(extra->getOperand(i + 1)); break;
    }
  }
}

void ReadResources(const llvm::MDNode *list, ResourceClass cls, std::vector<Resource> &out) {
  if (!list)
    return;
  for (auto &op : list->operands()) {
    auto m = Node(op);
    if (!m || m->getNumOperands() < 7)
      continue;
    Resource r;
    r.cls = cls;
    r.id = Int(m->getOperand(0));
    r.space = Int(m->getOperand(3));
    r.lower_bound = Int(m->getOperand(4));
    r.size = (uint32_t)Int(m->getOperand(5)); // -1 (unbounded) becomes ~0u
    switch (cls) {
    case ResourceClass::SRV: // ..., i32 kind, i32 sampleCount, !extra
      r.kind = ResourceKind(Int(m->getOperand(6)));
      if (m->getNumOperands() > 8) ReadResourceExtra(Node(m->getOperand(8)), r);
      break;
    case ResourceClass::UAV: // ..., i32 kind, i1 globallyCoherent, i1 hasCounter, i1 ROV, !extra
      r.kind = ResourceKind(Int(m->getOperand(6)));
      r.globally_coherent = Int(m->getOperand(7)) != 0;
      r.has_counter = Int(m->getOperand(8)) != 0;
      r.rasterizer_ordered = Int(m->getOperand(9)) != 0;
      if (m->getNumOperands() > 10) ReadResourceExtra(Node(m->getOperand(10)), r);
      break;
    case ResourceClass::CBuffer: // ..., i32 sizeInBytes, !extra
      r.kind = ResourceKind::CBuffer;
      r.cbuffer_size = Int(m->getOperand(6));
      break;
    case ResourceClass::Sampler: // ..., i32 samplerType (1 = comparison), !extra
      r.kind = ResourceKind::Sampler;
      r.comparison_sampler = Int(m->getOperand(6)) == 1;
      break;
    }
    out.push_back(r);
  }
}

} // namespace

llvm::Expected<std::unique_ptr<llvm::Module>>
LoadModule(llvm::LLVMContext &context, const char *bitcode, size_t size) {
  auto module = llvm::parseBitcodeFile(llvm::MemoryBufferRef(llvm::StringRef(bitcode, size), "dxil"), context);
  if (!module)
    return llvm::make_error<UnsupportedFeature>("DXIL: can't read the bitcode: " + llvm::toString(module.takeError()));
  return std::move(*module);
}

llvm::Expected<EntryInfo>
ReadEntry(const llvm::Module &module) {
  EntryInfo entry;
  auto shader_model = module.getNamedMetadata("dx.shaderModel");
  if (!shader_model || !shader_model->getNumOperands())
    return Missing("dx.shaderModel");
  auto sm = shader_model->getOperand(0);
  std::string stage = String(sm->getOperand(0));
  entry.kind = stage == "ps"   ? ShaderKind::Pixel
               : stage == "vs" ? ShaderKind::Vertex
               : stage == "gs" ? ShaderKind::Geometry
               : stage == "hs" ? ShaderKind::Hull
               : stage == "ds" ? ShaderKind::Domain
               : stage == "cs" ? ShaderKind::Compute
               : stage == "lib" ? ShaderKind::Library
                                : ShaderKind::Invalid;
  entry.sm_major = Int(sm->getOperand(1));
  entry.sm_minor = Int(sm->getOperand(2));

  auto entry_points = module.getNamedMetadata("dx.entryPoints");
  if (!entry_points || !entry_points->getNumOperands())
    return Missing("dx.entryPoints");
  // !{void ()* @main, !"main", !sigs, !resources, !props}
  auto ep = entry_points->getOperand(0);
  if (ep->getNumOperands() < 5)
    return Missing("entry point fields");
  auto fn = llvm::mdconst::dyn_extract_or_null<llvm::Function>(ep->getOperand(0));
  if (!fn)
    return Missing("entry function");
  entry.name = fn->getName().str();
  if (auto sigs = Node(ep->getOperand(2))) {
    entry.inputs = ReadSignature(Node(sigs->getOperand(0)));
    entry.outputs = ReadSignature(Node(sigs->getOperand(1)));
  }
  if (auto res = Node(ep->getOperand(3))) {
    ReadResources(Node(res->getOperand(0)), ResourceClass::SRV, entry.resources);
    ReadResources(Node(res->getOperand(1)), ResourceClass::UAV, entry.resources);
    ReadResources(Node(res->getOperand(2)), ResourceClass::CBuffer, entry.resources);
    ReadResources(Node(res->getOperand(3)), ResourceClass::Sampler, entry.resources);
  }
  if (auto props = Node(ep->getOperand(4)))
    for (unsigned i = 0; i + 1 < props->getNumOperands(); i += 2)
      if (Int(props->getOperand(i)) == 4) // NumThreads
        if (auto n = Node(props->getOperand(i + 1)))
          for (unsigned d = 0; d < 3 && d < n->getNumOperands(); d++)
            entry.numthreads[d] = Int(n->getOperand(d));
  return entry;
}

uint32_t
OpCode(const llvm::CallInst &call) {
  auto fn = call.getCalledFunction();
  if (!fn || !fn->getName().startswith("dx.op.") || call.arg_size() == 0)
    return ~0u;
  auto c = llvm::dyn_cast<llvm::ConstantInt>(call.getArgOperand(0));
  return c ? (uint32_t)c->getZExtValue() : ~0u;
}

bool
ModuleCallsOp(const llvm::Module &module, uint32_t opcode) {
  for (auto &fn : module)
    if (fn.getName().startswith("dx.op."))
      for (auto *user : fn.users())
        if (auto call = llvm::dyn_cast<llvm::CallInst>(user))
          if (OpCode(*call) == opcode)
            return true;
  return false;
}

} // namespace dxmt::dxil
