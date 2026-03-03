# instrument-amdgpu-kernels Architecture

## Overview
LLVM pass plugins for instrumenting AMDGPU kernels at the IR level. Creates instrumented clones of kernels that submit runtime data to dh_comms buffers.

**Recent Changes** (2026-03-03):
- Removed 5 unused plugins that don't use dh_comms infrastructure
- Removed `examples/`, `instrumentation/`, and `tests/` directories
- Renamed `lib/` to `src/` for clarity
- Simplified to 3 core plugins: AddressMessages, BBStart, BBInterval

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

All plugins use dh_comms device bitcode for message submission:

| Plugin | Source | Instruments | dh_comms Calls |
|--------|--------|-------------|----------------|
| AMDGCNSubmitAddressMessages | `src/AMDGCNSubmitAddressMessages.cpp` | Load/Store | `v_submit_address()` |
| AMDGCNSubmitBBStart | `src/AMDGCNSubmitBBStart.cpp` | Basic blocks | `s_submit_wave_header()` |
| AMDGCNSubmitBBInterval | `src/AMDGCNSubmitBBInterval.cpp` | BB timing | `s_submit_time_interval()` |

### Removed Plugins (2026-03-03)

The following plugins were removed as they don't integrate with dh_comms:
- `AMDGCNMemTrace` — used external instrumentation file instead of dh_comms
- `AMDGCNNumCacheLines` — used external instrumentation kernel
- `InjectAMDGCNFunction` — example/demo code for custom function injection
- `InjectAMDGCNInlineASM` — example/demo code for inline assembly
- `InjectAMDGCNSharedMemTtrace` — shared memory tracing (no dh_comms integration)

## Shared Infrastructure

| File | Purpose |
|------|---------|
| `src/InstrumentationCommon.cpp` | Kernel cloning, bitcode linking |
| `include/InstrumentationCommon.h` | Common API |

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

## Directory Structure

```
instrument-amdgpu-kernels/
├── src/               # Plugin source files (renamed from lib/)
│   ├── AMDGCNSubmitAddressMessages.cpp
│   ├── AMDGCNSubmitBBStart.cpp
│   ├── AMDGCNSubmitBBInterval.cpp
│   ├── InstrumentationCommon.cpp
│   └── CMakeLists.txt
├── include/           # Public headers
│   ├── AMDGCNSubmitAddressMessage.h
│   ├── AMDGCNSubmitBBStart.h
│   ├── AMDGCNSubmitBBInterval.h
│   ├── InstrumentationCommon.h
│   └── utils.h
└── CMakeLists.txt     # Root build file
```

## Last Verified
Commit: <pending commit>
Date: 2026-03-03
