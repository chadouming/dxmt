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
#include "dxil_types.hpp"
#include "../airconv_public.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <optional>

namespace dxmt::dxbc {
class SM50ShaderInternal;
}

// The DXIL front end's units, as airconv (dxbc_converter.cpp) and each other use them.
namespace dxmt::dxil {

struct Container {
  const char *bitcode;
  size_t bitcode_size;
};

// dxil_container.cpp: the DXIL part's bitcode; std::nullopt for a DXBC container; an error for a malformed DXIL part.
llvm::Expected<std::optional<Container>> FindDXIL(const void *bytecode, size_t size);

// dxil_metadata.cpp: parses bitcode into `context` (the caller chooses typed pointers), and reads the entry point.
llvm::Expected<std::unique_ptr<llvm::Module>> LoadModule(llvm::LLVMContext &context, const char *bitcode, size_t size);
llvm::Expected<EntryInfo> ReadEntry(const llvm::Module &module);
// The opcode of a dx.op call (its first argument), or ~0u for any other call.
uint32_t OpCode(const llvm::CallInst &call);
// Whether any dx.op call in the module has this opcode.
bool ModuleCallsOp(const llvm::Module &module, uint32_t opcode);

// dxil_initialize.cpp: SM50Initialize's DXIL branch.
llvm::Error InitializeDXIL(const Container &container, dxbc::SM50ShaderInternal *shader, MTL_SHADER_REFLECTION *refl);

// dxil_signature.cpp: pushes the signature handlers (entry point inputs/outputs) for this stage.
llvm::Error AddSignatureHandlers(const EntryInfo &entry, const llvm::Module &module, dxbc::SM50ShaderInternal *shader);

// dxil_converter.cpp: SM50Compile's DXIL branch; returns the AIR module, ready for airconv's passes.
llvm::Expected<std::unique_ptr<llvm::Module>> ConvertDXIL(
    dxbc::SM50ShaderInternal *shader, const char *name, llvm::LLVMContext &context,
    SM50_SHADER_COMPILATION_ARGUMENT_DATA *pArgs
);

} // namespace dxmt::dxil
