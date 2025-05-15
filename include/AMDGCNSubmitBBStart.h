#pragma once

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

namespace {

struct AMDGCNSubmitBBStart : public PassInfoMixin<AMDGCNSubmitBBStart> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    bool Changed = runOnModule(M);

    return (Changed ? llvm::PreservedAnalyses::none()
                    : llvm::PreservedAnalyses::all());
  }
  bool runOnModule(llvm::Module &M);
  // isRequired being set to true keeps this pass from being skipped
  // if it has the optnone LLVM attribute
  static bool isRequired() { return true; }
};

} // end anonymous namespace
