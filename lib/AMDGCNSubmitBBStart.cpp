#include "AMDGCNSubmitBBStart.h"
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

void InjectInstrumentationFunction(const BasicBlock::iterator &I,
                                   const Function &F, llvm::Module &M,
                                   uint32_t &LocationCounter, llvm::Value *Ptr,
                                   bool PrintLocationInfo) {
  DILocation *DL = dyn_cast<Instruction>(I)->getDebugLoc();

  std::string dbgFile =
      DL != nullptr ? getFullPath(DL) : "<unknown source file>";
  size_t dbgFileHash = std::hash<std::string>{}(dbgFile);

  IRBuilder<> Builder(dyn_cast<Instruction>(I));
  Value *DbgFileHashVal = Builder.getInt64(dbgFileHash);
  Value *DbgLineVal = Builder.getInt32(DL != nullptr ? DL->getLine() : 0);
  Value *DbgColumnVal = Builder.getInt32(DL != nullptr ? DL->getColumn() : 0);

  std::string SourceInfo = (F.getName() + "     " + dbgFile + ":" +
                            Twine(DL != nullptr ? DL->getLine() : 0) + ":" +
                            Twine(DL != nullptr ? DL->getColumn() : 0))
                               .str();

  auto &CTX = M.getContext();
  FunctionType *FT =
      FunctionType::get(Type::getVoidTy(CTX),
                        {Ptr->getType(), Type::getInt64Ty(CTX),
                         Type::getInt32Ty(CTX), Type::getInt32Ty(CTX)},
                        false);
  FunctionCallee InstrumentationFunction =
      M.getOrInsertFunction("s_submit_wave_header", FT);
  Builder.CreateCall(FT, cast<Function>(InstrumentationFunction.getCallee()),
                     {Ptr, DbgFileHashVal, DbgLineVal, DbgColumnVal});
  if (PrintLocationInfo) {
    errs() << "Injecting Basic Block tracker " << LocationCounter
           << " into AMDGPU Kernel: " << SourceInfo << "\n";
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

bool AMDGCNSubmitBBStart::runOnModule(Module &M) {
  errs() << "Running AMDGCNSubmitBBStart on module: " << M.getName() << "\n";
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
      auto I = BB->begin();
      if (I != BB->end()) {
        InjectInstrumentationFunction(I, *NF, M, LocationCounter, bufferPtr,
                                      true);
        ModifiedCodeGen = true;
      }
    }
  }
  errs() << "Done running AMDGCNSubmitBBStart on module: " << M.getName()
         << "\n";
  return ModifiedCodeGen;
}

PassPluginLibraryInfo getPassPluginInfo() {
  const auto callback = [](PassBuilder &PB) {
    PB.registerOptimizerLastEPCallback(
        [&](ModulePassManager &MPM, auto &&...args) {
          MPM.addPass(AMDGCNSubmitBBStart());
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
