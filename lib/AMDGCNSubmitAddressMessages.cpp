
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
#include "AMDGCNSubmitAddressMessage.h"
#include "utils.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <iostream>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstdlib>
#include <dlfcn.h>
#include <limits.h>
#include <type_traits>
#include <unistd.h>

using namespace llvm;
using namespace std;

std::map<int, std::string> AddrSpaceMap = {
    {0, "FLAT"}, {1, "GLOBAL"}, {3, "SHARED"}, {4, "CONSTANT"}};

std::string LoadOrStoreMap(const BasicBlock::iterator &I) {
  if (dyn_cast<LoadInst>(I) != nullptr)
    return "LOAD";
  else if (dyn_cast<StoreInst>(I) != nullptr)
    return "STORE";
  else
    throw std::runtime_error("Error: unknown operation type");
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

template <typename LoadOrStoreInst>
void InjectInstrumentationFunction(const BasicBlock::iterator &I,
                                   const Function &F, llvm::Module &M,
                                   uint32_t &LocationCounter, llvm::Value *Ptr,
                                   bool PrintLocationInfo) {
  auto &CTX = M.getContext();
  auto LSI = dyn_cast<LoadOrStoreInst>(I);
  Value *AccessTypeVal;
  Type *PointeeType;
  IRBuilder<> Builder(dyn_cast<Instruction>(I));
  auto LI = dyn_cast<LoadInst>(I);
  auto SI = dyn_cast<StoreInst>(I);
  if (LI) {
    AccessTypeVal = Builder.getInt8(0b01);
    PointeeType = LI->getType();
  } else if (SI) {
    AccessTypeVal = Builder.getInt8(0b10);
    PointeeType = SI->getValueOperand()->getType();
  } else {
    return;
  }
  if (not LSI)
    return;

  DILocation *DL = dyn_cast<Instruction>(I)->getDebugLoc();

  std::string dbgFile =
      DL != nullptr ? getFullPath(DL) : "<unknown source file>";
  size_t dbgFileHash = std::hash<std::string>{}(dbgFile);

  Value *Addr = LSI->getPointerOperand();
  Value *DbgFileHashVal = Builder.getInt64(dbgFileHash);
  Value *DbgLineVal = Builder.getInt32(DL != nullptr ? DL->getLine() : 0);
  Value *DbgColumnVal = Builder.getInt32(DL != nullptr ? DL->getColumn() : 0);
  Value *Op = LSI->getPointerOperand()->stripPointerCasts();
  uint32_t AddrSpace = cast<PointerType>(Op->getType())->getAddressSpace();
  Value *AddrSpaceVal = Builder.getInt8(AddrSpace);
  uint16_t PointeeTypeSize = M.getDataLayout().getTypeStoreSize(PointeeType);
  Value *PointeeTypeSizeVal = Builder.getInt16(PointeeTypeSize);

  std::string SourceInfo = (F.getName() + "     " + dbgFile + ":" +
                            Twine(DL != nullptr ? DL->getLine() : 0) + ":" +
                            Twine(DL != nullptr ? DL->getColumn() : 0))
                               .str();

  // v_submit_message expects addresses (passed in Addr) to be 64-bits. However,
  // LDS pointers are 32 bits, so we have to cast those. Ptr (the pointer to the
  // dh_comms resources in global device memory) is 64-bits, so we use its type
  // to do the cast.

  Value *Addr64 = Builder.CreatePointerCast(Addr, Ptr->getType());

  FunctionType *FT = FunctionType::get(
      Type::getVoidTy(CTX),
      {Ptr->getType(), Addr64->getType(), Type::getInt64Ty(CTX),
       Type::getInt32Ty(CTX), Type::getInt32Ty(CTX), Type::getInt8Ty(CTX),
       Type::getInt8Ty(CTX), Type::getInt16Ty(CTX)},
      false);
  FunctionCallee InstrumentationFunction =
      M.getOrInsertFunction("v_submit_address", FT);
  Builder.CreateCall(FT, cast<Function>(InstrumentationFunction.getCallee()),
                     {Ptr, Addr64, DbgFileHashVal, DbgLineVal, DbgColumnVal,
                      AccessTypeVal, AddrSpaceVal, PointeeTypeSizeVal});
  if (PrintLocationInfo) {
    errs() << "Injecting Mem Trace Function Into AMDGPU Kernel: " << SourceInfo
           << "\n";
    errs() << LocationCounter << "     " << SourceInfo << "     "
           << AddrSpaceMap[AddrSpace] << "     " << LoadOrStoreMap(I) << "\n";
  }
  LocationCounter++;
}

std::string getBitcodePath() {
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void *>(&getBitcodePath), &dl_info) == 0) {
    errs() << "Error: Could not determine IR pass plugin path!\n";
    return "";
  }

  std::string PluginPath = dl_info.dli_fname;
  size_t LastSlash = PluginPath.find_last_of('/');
  if (LastSlash == std::string::npos) {
    errs() << "Error: IR pass plugin path invalid!\n";
    return "";
  }

  errs() << "IR pass plugin path: " << PluginPath << "\n";

  std::string PluginDir = PluginPath.substr(0, LastSlash); // Extract directory
  if (PluginDir.empty()) {
    errs() << "Error: Could not determine plugin directory!\n";
    return "";
  }
  if (PluginPath.find("/lib/") != std::string::npos) {
    PluginDir = PluginDir.substr(0, PluginDir.find("/lib/"));
  }

  std::string CodeObjectVersion =
      (PluginPath.find("triton") != std::string::npos) ? "co4" : "co5";
  std::string BitcodePath =
      PluginDir + "/dh_comms_dev_" + CodeObjectVersion + ".bc";

  return BitcodePath;
}

bool AMDGCNSubmitAddressMessage::runOnModule(Module &M) {
  errs() << "Running AMDGCNSubmitAddressMessage on module: " << M.getName()
         << "\n";
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
    errs() << "device function module found for " << TripleStr << "\n";
  } else { // Not an AMDGPU target
    errs() << TripleStr << ": Not an AMDGPU target, skipping pass.\n";
    return false;
  }

  std::string BitcodePath = getBitcodePath();

  if (!llvm::sys::fs::exists(BitcodePath)) {
    errs() << "Error: Bitcode file not found at " << BitcodePath << "\n";
    return false;
  }

  auto Buffer = MemoryBuffer::getFile(BitcodePath);
  if (!Buffer) {
    errs() << "Error loading bitcode file: " << BitcodePath << "\n";
    return false;
  }

  auto DeviceModuleOrErr =
      parseBitcodeFile(Buffer->get()->getMemBufferRef(), M.getContext());
  if (!DeviceModuleOrErr) {
    errs() << "Error parsing bitcode file: " << BitcodePath << "\n";
    return false;
  }

  std::unique_ptr<llvm::Module> DeviceModule =
      std::move(DeviceModuleOrErr.get());

  errs() << "Linking device module from " << BitcodePath
         << " into GPU module\n";
  if (llvm::Linker::linkModules(M, std::move(DeviceModule))) {
    errs() << "Error linking device function module into instrumented "
              "module!\n";
    return false;
  }

  // Now v_submit_address should be available inside M

  std::vector<Function *> GpuKernels;

  for (auto &F : M) {
    if (F.isIntrinsic())
      continue;
    if (F.getCallingConv() == CallingConv::AMDGPU_KERNEL) {
      GpuKernels.push_back(&F);
    }
  }

  bool ModifiedCodeGen = false;
  for (auto &I : GpuKernels) {
    std::string AugmentedName = "__amd_crk_" + I->getName().str() + "Pv";
    ValueToValueMapTy VMap;
    // Add an extra ptr arg on to the instrumented kernels
    std::vector<Type *> ArgTypes;
    for (auto arg = I->arg_begin(); arg != I->arg_end(); ++arg) {
      ArgTypes.push_back(arg->getType());
    }
    ArgTypes.push_back(PointerType::get(M.getContext(), /*AddrSpace=*/0));
    FunctionType *FTy =
        FunctionType::get(I->getFunctionType()->getReturnType(), ArgTypes,
                          I->getFunctionType()->isVarArg());
    Function *NF = Function::Create(FTy, I->getLinkage(), I->getAddressSpace(),
                                    AugmentedName, &M);
    NF->copyAttributesFrom(I);
    VMap[I] = NF;

    // Get the ptr we just added to the kernel arguments
    Value *bufferPtr = &*NF->arg_end() - 1;
    Function *F = cast<Function>(VMap[I]);

    Function::arg_iterator DestI = F->arg_begin();
    for (const Argument &J : I->args()) {
      DestI->setName(J.getName());
      VMap[&J] = &*DestI++;
    }
    SmallVector<ReturnInst *, 8> Returns; // Ignore returns cloned.
    CloneFunctionInto(F, I, VMap, CloneFunctionChangeType::GlobalChanges,
                      Returns);
    uint32_t LocationCounter = 0;
    for (Function::iterator BB = NF->begin(); BB != NF->end(); BB++) {
      for (BasicBlock::iterator I = BB->begin(); I != BB->end(); I++) {
        if (dyn_cast<LoadInst>(I) != nullptr) {
          InjectInstrumentationFunction<LoadInst>(I, *NF, M, LocationCounter,
                                                  bufferPtr, true);
          ModifiedCodeGen = true;
        } else if (dyn_cast<StoreInst>(I) != nullptr) {
          InjectInstrumentationFunction<StoreInst>(I, *NF, M, LocationCounter,
                                                   bufferPtr, true);
          ModifiedCodeGen = true;
        }
      }
    }
  }
  errs() << "Done running AMDGCNSubmitAddressMessage on module: " << M.getName()
         << "\n";
  return ModifiedCodeGen;
}

PassPluginLibraryInfo getPassPluginInfo() {
  const auto callback = [](PassBuilder &PB) {
    PB.registerOptimizerLastEPCallback(
        [&](ModulePassManager &MPM, auto &&...args) {
          MPM.addPass(AMDGCNSubmitAddressMessage());
          return true;
        });
  };

  return {LLVM_PLUGIN_API_VERSION, "amdgcn-submit-address-message",
          LLVM_VERSION_STRING, callback};
};

extern "C"
    //    __attribute__((visibility("default"))) PassPluginLibraryInfo extern
    //    "C"
    LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo
    llvmGetPassPluginInfo() {
  return getPassPluginInfo();
}
