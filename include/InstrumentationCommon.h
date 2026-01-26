/******************************************************************************
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*******************************************************************************/
#ifndef INSTRUMENTATION_COMMON_H
#define INSTRUMENTATION_COMMON_H

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <memory>
#include <string>
#include <vector>

namespace llvm {
class DILocation;
}

namespace instrumentation {
namespace common {

// Get the path to the bitcode file containing device functions for
// instrumentation based on the module's architecture and code object version
std::string getBitcodePath(const llvm::Module &M);

// Extract full file path from debug location information
std::string getFullPath(const llvm::DILocation *DIL);

// Validate that the module targets AMDGPU (amdgcn-amd-amdhsa)
// Returns true if valid AMDGPU target, false otherwise
bool validateAMDGPUTarget(const llvm::Module &M);

// Load bitcode file and link it into the module
// Automatically determines the bitcode path based on the module's architecture
// Returns the loaded device module on success, nullptr on failure
std::unique_ptr<llvm::Module> loadAndLinkBitcode(llvm::Module &M);

// Collect all GPU kernel functions from the module
std::vector<llvm::Function *> collectGPUKernels(llvm::Module &M);

// Clone a kernel function with an additional pointer argument for
// instrumentation buffer. Creates a new function with name
// "__amd_crk_<original_name>Pv"
llvm::Function *cloneKernelWithExtraArg(llvm::Function *OrigKernel,
                                        llvm::Module &M,
                                        llvm::ValueToValueMapTy &VMap);

} // namespace common
} // namespace instrumentation

#endif // INSTRUMENTATION_COMMON_H
