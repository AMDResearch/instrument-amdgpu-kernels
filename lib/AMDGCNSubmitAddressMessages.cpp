
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
#include "InstrumentationCommon.h"
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
using namespace instrumentation::common;

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

bool AMDGCNSubmitAddressMessage::runOnModule(Module &M) {
  errs() << "Running AMDGCNSubmitAddressMessage on module: " << M.getName()
         << "\n";

  if (!validateAMDGPUTarget(M)) {
    return false;
  }

  if (!loadAndLinkBitcode(M)) {
    return false;
  }

  // Now v_submit_address should be available inside M

  std::vector<Function *> GpuKernels = collectGPUKernels(M);

  bool ModifiedCodeGen = false;
  for (auto &I : GpuKernels) {
    ValueToValueMapTy VMap;
    Function *NF = cloneKernelWithExtraArg(I, M, VMap);

    // Get the ptr we just added to the kernel arguments
    Value *bufferPtr = &*NF->arg_end() - 1;
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
    LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getPassPluginInfo();
}
