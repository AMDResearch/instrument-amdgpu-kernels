#include "AMDGCNSubmitBBInterval.h"
#include "utils.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <limits.h>
#include <type_traits>
#include <unistd.h>
#include <vector>

using namespace llvm;
using namespace std;

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

bool AMDGCNSubmitBBInterval::runOnModule(Module &M) {
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
    errs() << "Running AMDGCNSubmitBBInterval on " << TripleStr
           << " device module for " << M.getName() << "\n";
  } else { // Not an AMDGPU target
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

  if (llvm::Linker::linkModules(M, std::move(DeviceModule))) {
    errs() << "Error linking device function module into instrumented "
              "module!\n";
    return false;
  }

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
    // Instrument each basic block in the cloned kernel NF:
    unsigned bbIndex = 0;
    for (auto BB = NF->begin(); BB != NF->end(); ++BB) {
      Instruction *firstInst = nullptr;
      if (!BB->empty()) {
        for (auto &Inst : *BB) {
          if (!isa<PHINode>(Inst)) {
            firstInst = &Inst;
            break;
          }
        }
      }
      IRBuilder<> BuilderStart(firstInst ? firstInst : &*BB->begin());
      // Obtain (or declare) the s_clock64() function from dh_comms_dev.h
      Function *sClock64Func = cast<Function>(
          M.getOrInsertFunction(
               "s_clock64",
               FunctionType::get(BuilderStart.getInt64Ty(), {}, false))
              .getCallee());
      // Insert call to s_clock64() at BB entry (replacing __clock64())
      CallInst *startCall = BuilderStart.CreateCall(sClock64Func, {});
      // Allocate a 2-element array for timestamps and get pointer with proper
      // GEP
      AllocaInst *timeIntervalAlloca = BuilderStart.CreateAlloca(
          ArrayType::get(BuilderStart.getInt64Ty(), 2), nullptr,
          "timeInterval");
      Value *zero32 = BuilderStart.getInt32(0);
      Value *startPtr = BuilderStart.CreateInBoundsGEP(
          timeIntervalAlloca->getAllocatedType(), timeIntervalAlloca,
          {BuilderStart.getInt32(0), zero32});
      BuilderStart.CreateStore(startCall, startPtr);

      // Compute debug info constants based on the first instruction (or
      // defaults)
      uint64_t fileHashConst;
      uint32_t lineConst;
      uint32_t columnConst;
      if (firstInst) {
        DILocation *DL = firstInst->getDebugLoc();
        if (DL) {
          std::string dbgFile = getFullPath(DL);
          fileHashConst = std::hash<std::string>{}(dbgFile);
          lineConst = DL->getLine();
          columnConst = DL->getColumn();
        } else {
          fileHashConst = 0;
          lineConst = 0xffffff;
          columnConst = 0xffffff;
        }
      } else {
        fileHashConst = 0;
        lineConst = 0xffffff;
        columnConst = 0xffffff;
      }

      // Insert s_clock64() call at BB end (before terminator)
      Instruction *insertPt = BB->getTerminator();
      IRBuilder<> BuilderEnd(insertPt);
      CallInst *endCall = BuilderEnd.CreateCall(sClock64Func, {});
      Value *one32 = BuilderEnd.getInt32(1);
      Value *endPtr = BuilderEnd.CreateInBoundsGEP(
          timeIntervalAlloca->getAllocatedType(), timeIntervalAlloca,
          {BuilderEnd.getInt32(0), one32});
      BuilderEnd.CreateStore(endCall, endPtr);

      // Prepare constant values for debug metadata
      Value *dbgFileHashVal = BuilderEnd.getInt64(fileHashConst);
      Value *dbgLineVal = BuilderEnd.getInt32(lineConst);
      Value *dbgColumnVal = BuilderEnd.getInt32(columnConst);

      // Get (or insert) the declaration for s_submit_time_interval
      Type *VoidPtrTy = PointerType::get(Type::getInt8Ty(M.getContext()), 0);
      Function *sSubmitTimeInterval = cast<Function>(
          M.getOrInsertFunction(
               "s_submit_time_interval",
               FunctionType::get(Type::getVoidTy(M.getContext()),
                                 {bufferPtr->getType(), VoidPtrTy,
                                  Type::getInt64Ty(M.getContext()),
                                  Type::getInt32Ty(M.getContext()),
                                  Type::getInt32Ty(M.getContext()),
                                  Type::getInt32Ty(M.getContext())},
                                 false))
              .getCallee());
      // Cast the time interval allocation to a void Ptr
      Value *timeIntervalPtr =
          BuilderEnd.CreatePointerCast(timeIntervalAlloca, VoidPtrTy);

      // Insert call to s_submit_time_interval with debug info and the timing
      // struct
      BuilderEnd.CreateCall(sSubmitTimeInterval,
                            {bufferPtr, timeIntervalPtr, dbgFileHashVal,
                             dbgLineVal, dbgColumnVal,
                             BuilderEnd.getInt32(bbIndex)});

      bbIndex++;
      ModifiedCodeGen = true;
    }
    errs() << "AMDGCNSubmitBBInterval: instrumented " << bbIndex
           << " basic blocks for kernel " << I->getName() << "\n";
  }
  errs() << "Done running AMDGCNSubmitBBInterval on " << M.getName() << "\n";
  return ModifiedCodeGen;
}

PassPluginLibraryInfo getPassPluginInfo() {
  const auto callback = [](PassBuilder &PB) {
    PB.registerOptimizerLastEPCallback(
        [&](ModulePassManager &MPM, auto &&...args) {
          MPM.addPass(AMDGCNSubmitBBInterval());
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
