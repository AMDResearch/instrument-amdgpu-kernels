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
#include "InstrumentationCommon.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <cstdlib>
#include <dlfcn.h>
#include <type_traits>

using namespace llvm;

namespace instrumentation {
namespace common {

std::string getBitcodePath(const llvm::Module &M) {
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void *>(&getBitcodePath), &dl_info) == 0) {
    llvm::errs() << "Error: Could not determine IR pass plugin path!\n";
    return "";
  }

  std::string PluginPath = dl_info.dli_fname;
  size_t LastSlash = PluginPath.find_last_of('/');
  if (LastSlash == std::string::npos) {
    llvm::errs() << "Error: IR pass plugin path invalid!\n";
    return "";
  }

  llvm::errs() << "IR pass plugin path: " << PluginPath << "\n";

  std::string PluginDir = PluginPath.substr(0, LastSlash); // Extract directory
  if (PluginDir.empty()) {
    llvm::errs() << "Error: Could not determine plugin directory!\n";
    return "";
  }
  if (PluginPath.find("/lib/") != std::string::npos) {
    PluginDir = PluginDir.substr(0, PluginDir.find("/lib/"));
  }

  std::string CodeObjectVersion =
      (PluginPath.find("triton") != std::string::npos) ? "_co5" : "_co6";

  // Determine CDNAVersion based on architecture
  std::string CDNAVersion;
  // Try to get the target-cpu from the module flag
  std::string arch;
  if (auto *cpuMD =
          llvm::cast_or_null<llvm::MDString>(M.getModuleFlag("target-cpu"))) {
    arch = cpuMD->getString().str();
  } else {
    // Fallback: try to get from a kernel function attribute
    for (const auto &F : M) {
      if (F.hasFnAttribute("target-cpu")) {
        arch = F.getFnAttribute("target-cpu").getValueAsString().str();
        break;
      }
    }
  }

  if (arch != "") {
    llvm::errs() << "Detected architecture: " << arch << "\n";
  } else {
    llvm::errs() << "Warning: Could not determine target architecture, "
                    "defaulting to cdna2.\n";
    arch = "unknown";
  }

  if (arch == "gfx940" || arch == "gfx941" || arch == "gfx942") {
    CDNAVersion = "_cdna3";
  } else {
    CDNAVersion = "_cdna2";
  }

  std::string BitcodePath =
      PluginDir + "/dh_comms_dev" + CDNAVersion + CodeObjectVersion + ".bc";

  return BitcodePath;
}

std::string getFullPath(const llvm::DILocation *DIL) {
  if (!DIL)
    return "";

  const llvm::DIFile *File = DIL->getScope()->getFile();
  if (!File)
    return "";

  std::string Directory = File->getDirectory().str();
  std::string FileName = File->getFilename().str();

  if (!Directory.empty())
    return Directory + "/" + FileName; // Concatenate full path
  else
    return FileName; // No directory available, return just the file name
}

bool validateAMDGPUTarget(const llvm::Module &M) {
  auto TargetTriple = M.getTargetTriple();

  // Use std::string comparison if needed, otherwise call str()
  std::string TripleStr = [](const auto &T) -> std::string {
    if constexpr (std::is_same_v<std::decay_t<decltype(T)>, std::string>) {
      return T; // Already a std::string
    } else {
      return T.str(); // Convert llvm::Triple to std::string
    }
  }(TargetTriple);

  if (TripleStr == "amdgcn-amd-amdhsa") {
    llvm::errs() << "device function module found for " << TripleStr << "\n";
    return true;
  } else { // Not an AMDGPU target
    llvm::errs() << TripleStr << ": Not an AMDGPU target, skipping pass.\n";
    return false;
  }
}

std::unique_ptr<llvm::Module> loadAndLinkBitcode(llvm::Module &M) {
  std::string BitcodePath = getBitcodePath(M);

  if (!llvm::sys::fs::exists(BitcodePath)) {
    llvm::errs() << "Error: Bitcode file not found at " << BitcodePath << "\n";
    return nullptr;
  }

  auto Buffer = MemoryBuffer::getFile(BitcodePath);
  if (!Buffer) {
    llvm::errs() << "Error loading bitcode file: " << BitcodePath << "\n";
    return nullptr;
  }

  auto DeviceModuleOrErr =
      parseBitcodeFile(Buffer->get()->getMemBufferRef(), M.getContext());
  if (!DeviceModuleOrErr) {
    llvm::errs() << "Error parsing bitcode file: " << BitcodePath << "\n";
    return nullptr;
  }

  std::unique_ptr<llvm::Module> DeviceModule =
      std::move(DeviceModuleOrErr.get());

  llvm::errs() << "Linking device module from " << BitcodePath
               << " into GPU module\n";
  if (llvm::Linker::linkModules(M, CloneModule(*DeviceModule))) {
    llvm::errs() << "Error linking device function module into instrumented "
                    "module!\n";
    return nullptr;
  }

  return DeviceModule;
}

std::vector<llvm::Function *> collectGPUKernels(llvm::Module &M) {
  std::vector<llvm::Function *> GpuKernels;

  for (auto &F : M) {
    if (F.isIntrinsic())
      continue;
    if (F.getCallingConv() == CallingConv::AMDGPU_KERNEL) {
      GpuKernels.push_back(&F);
    }
  }

  return GpuKernels;
}

llvm::Function *cloneKernelWithExtraArg(llvm::Function *OrigKernel,
                                        llvm::Module &M,
                                        llvm::ValueToValueMapTy &VMap) {
  std::string AugmentedName = "__amd_crk_" + OrigKernel->getName().str() + "Pv";

  // Add an extra ptr arg on to the instrumented kernels
  std::vector<Type *> ArgTypes;
  for (auto arg = OrigKernel->arg_begin(); arg != OrigKernel->arg_end();
       ++arg) {
    ArgTypes.push_back(arg->getType());
  }
  ArgTypes.push_back(PointerType::get(M.getContext(), /*AddrSpace=*/0));

  FunctionType *FTy =
      FunctionType::get(OrigKernel->getFunctionType()->getReturnType(),
                        ArgTypes, OrigKernel->getFunctionType()->isVarArg());

  Function *NF =
      Function::Create(FTy, OrigKernel->getLinkage(),
                       OrigKernel->getAddressSpace(), AugmentedName, &M);
  NF->copyAttributesFrom(OrigKernel);
  VMap[OrigKernel] = NF;

  Function *F = cast<Function>(VMap[OrigKernel]);

  Function::arg_iterator DestI = F->arg_begin();
  for (const Argument &J : OrigKernel->args()) {
    DestI->setName(J.getName());
    VMap[&J] = &*DestI++;
  }

  SmallVector<ReturnInst *, 8> Returns; // Ignore returns cloned.
  CloneFunctionInto(F, OrigKernel, VMap, CloneFunctionChangeType::GlobalChanges,
                    Returns);

  return F;
}

} // namespace common
} // namespace instrumentation
