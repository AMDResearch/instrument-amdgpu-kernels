# instrument-amdgpu-kernels Architecture

## Overview
LLVM pass plugins for instrumenting AMDGPU kernels at the IR level. Creates instrumented clones of kernels that submit runtime data to dh_comms buffers.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLVM Compilation Pipeline                         │
│                                                                      │
│  Source → Frontend → IR → [Instrumentation Pass] → Backend → ISA    │
│                            ▲                                         │
│                            │                                         │
│                     LLVM_PASS_PLUGIN                                │
└────────────────────────────┼────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                    Pass Plugin (.so)                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  For each kernel:                                            │   │
│  │  1. Clone kernel → __amd_crk_<name>Pv                       │   │
│  │  2. Add void* arg (dh_comms_descriptor*)                    │   │
│  │  3. Link device bitcode (v_submit_* functions)              │   │
│  │  4. Insert instrumentation calls at load/store/etc          │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. LLVM loads plugin via `-fplugin=` or `LLVM_PASS_PLUGIN`
2. Plugin runs on each module after frontend
3. `collectGPUKernels()` finds all `amdgpu_kernel` functions
4. `cloneKernelWithExtraArg()` creates instrumented clone
5. Plugin-specific pass inserts instrumentation calls
6. Device bitcode linked in for `v_submit_*` implementations
7. Both original and instrumented kernels in output

## Key Invariants

- Original kernel preserved unchanged
- Instrumented clone: `__amd_crk_<original_name>Pv`
- Extra argument: `void* dh_comms_descriptor*` (last arg)
- DWARF info extracted from IR and embedded in messages
- Target must be `amdgcn-amd-amdhsa`

## Plugins

| Plugin | Source | Instruments | Message Type |
|--------|--------|-------------|--------------|
| AMDGCNSubmitAddressMessages | `lib/AMDGCNSubmitAddressMessages.cpp` | Load/Store | Address (64 per wave) |
| AMDGCNMemTrace | `lib/AMDGCNMemTrace.cpp` | Load/Store | Full trace |
| AMDGCNNumCacheLines | `lib/AMDGCNNumCacheLines.cpp` | Load/Store | Cache line count |
| AMDGCNSubmitBBStart | `lib/AMDGCNSubmitBBStart.cpp` | Basic blocks | BB entry |
| AMDGCNSubmitBBInterval | `lib/AMDGCNSubmitBBInterval.cpp` | Basic blocks | BB timing |

## Shared Infrastructure

| File | Purpose |
|------|---------|
| `lib/InstrumentationCommon.cpp` | Kernel cloning, bitcode linking |
| `include/InstrumentationCommon.h` | Common API |
| `instrumentation/*.cpp` | Device-side instrumentation kernels (compiled to bitcode) |

## Key Functions

### InstrumentationCommon
- `collectGPUKernels(M)` — find kernels in module — `InstrumentationCommon.h:56`
- `cloneKernelWithExtraArg(F, M, VMap)` — create instrumented clone — `InstrumentationCommon.h:61`
- `loadAndLinkBitcode(M)` — link device functions — `InstrumentationCommon.h:53`
- `getBitcodePath(M)` — find bitcode for arch — `InstrumentationCommon.h:41`
- `getFullPath(DIL)` — extract source path from debug info — `InstrumentationCommon.h:44`

## Address Space Mapping

```cpp
std::map<int, std::string> AddrSpaceMap = {
  {0, "FLAT"}, {1, "GLOBAL"}, {3, "SHARED"}, {4, "CONSTANT"}
};
```

## Usage

```bash
# ROCm LLVM
clang++ -fplugin=/path/to/libAMDGCNSubmitAddressMessages-rocm.so ...

# Triton LLVM
# Set LLVM_PASS_PLUGIN_PATH env var
```

## Build

- CMake-based
- Requires LLVM development headers
- Produces separate .so per plugin
- May build twice: once for ROCm LLVM, once for Triton LLVM

## Dependencies
- LLVM (PassPlugin API, IR manipulation)
- dh_comms device bitcode (linked at IR level)

## Known Limitations
- Must match LLVM version used by compiler
- Bitcode path resolution assumes standard installation layout

## Last Verified
Date: 2026-03-02
