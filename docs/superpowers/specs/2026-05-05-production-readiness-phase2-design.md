# Production Readiness Phase 2 — Design Spec

**Date**: 2026-05-05
**Goal**: Achieve full TeXmacs compatibility: load all 663 Scheme files (~238K lines) through pscm with a complete C API, enabling pscm to replace Guile 1.8 in TeXmacs.

## Current State

pscm v0.5.0 post-Phase-1: all 39 CTest tests pass, all lit suites pass (base, cont, macro, core, sicp). TeXmacs init-texmacs.scm loads successfully. Guile 1.8 feature audit: 396 present, 13 partial, 88 missing. Key known issues:

- Float comparison truncates doubles to int64_t (silently wrong results)
- GC has 4 abort() sites on heap/root exhaustion
- write/display truncates output at 4096 bytes
- 6 missing C API wrappers for module operations
- 88 feature gaps across built-ins, numerics, reader/printer, C API

## Architecture

```
                    +--------------------------+
                    |   C: Incremental Loading |
                    |   kernel -> utils ->     |
                    |   generic -> plugins     |
                    +----------+---------------+
                               | depends on
              +----------------+----------------+
              |                |                |
    +---------v------+  +-----v-------+  +-----v------+
    | A1: Float      |  | A2: GC      |  | A3: Print   |
    | Comparison Fix |  | abort() Fix |  | Buffer Fix  |
    +----------------+  +------------+  +------------+
              |                |                |
              +----------------+----------------+
                               |
                    +----------v---------------+
                    | B: C API Phase B         |
                    | 6 module wrappers        |
                    +--------------------------+
```

Track A (stability) and Track B (C API) are independent. Track C (incremental loading) depends on Track A completion.

## Track A: Stability Fixes

### A1: Float Comparison Truncation Fix

**File**: `src/c/number.cc`

**Problem**: `<`, `>`, `<=`, `>=` use a `BinaryOperator` template that truncates `double` to `int64_t` before comparison when both operands are FLOAT type. `(< -1.5 -1.0)` incorrectly returns `#f`.

**Fix**: Add a float-specific comparison path, similar to how `+`/`-`/`*`/`/` already use `needs_float_promotion`. When either operand is float, compare as `double` directly rather than truncating.

**Tests**: Add negative-float comparison cases to existing numeric test coverage.

### A2: GC abort() Elimination

**Files**: `src/c/gc.cc`, `src/c/gc.h`

**Problem**: 4 `abort()` calls on resource exhaustion:
1. Startup mmap failure — no recovery path, keep abort
2. Runtime mmap failure — should throw Scheme error instead
3. Segment pool full (MAX_SEGMENTS=16) — should throw Scheme error
4. Root registry full (MAX_ROOTS=2048) — should be dynamically expandable

**Fix**:
- Startup mmap (site 1): keep abort — can't recover from init failure
- Runtime mmap / segment pool (sites 2-3): throw `scm_throw("gc-error", ...)` to let callers handle exhaustion
- Root registry (site 4): replace fixed array with dynamic allocation (vector-based growth from initial 2048)

### A3: Print Buffer 4096-Byte Limit

**File**: `src/c/print.cc`

**Problem**: `write`/`display` redirect stdout to a temp file, then read back at most 4096 bytes. Larger output is silently truncated.

**Fix**: Use a dynamically growing buffer — allocate initial 4096, realloc and loop-read until EOF. No hard ceiling.

## Track B: C API Phase B

**Files**: `src/c/pscm_api.h`, `src/c/pscm_api.cc`

**Goal**: Expose module operations through the public C API.

**Missing wrappers**:

| Wrapper | Internal impl | Purpose |
|---------|-------------|---------|
| `pscm_c_resolve_module` | `scm_c_resolve_module` | Look up or create a module by name |
| `pscm_c_module_lookup` | `scm_c_module_lookup` | Look up a variable in a module |
| `pscm_c_module_define` | `scm_c_module_define` | Define a variable in a module |
| `pscm_c_use_module` | `scm_c_use_module` | Import a module's public interface |
| `pscm_c_current_module` | `scm_current_module` | Get the current module |
| `pscm_c_set_current_module` | `scm_set_current_module` | Set the current module |

Each is a thin wrapper: validate parameters, delegate to the internal scm_* function, return SCM*.

## Track C: Incremental Loading

### Core Loop

For each module directory: write load test → run → classify error → fix → commit → repeat.

### Error Classification

| Category | Example | Action |
|----------|---------|--------|
| Missing Scheme built-in | `unbound variable: read-line` | Implement in pscm |
| Syntax/macro mismatch | `quasiquote: bad syntax` | Analyze Guile 1.8 behavior, fix pscm |
| Missing C++ function | `unbound variable: get-output-tree` | Add Scheme stub (real impl later) |
| GC issue | Heap exhaustion | Return to Track A, expand limits |
| Semantic mismatch | Wrong return value | Analyze difference, fix pscm |

### Loading Order

```
Already passing      C1 target          C2 target          C3 target
-------------        ---------          ---------          ---------
init-texmacs.scm     kernel/library/*   utils/*            generic/*
boot.scm             kernel/regexp/*    utils/library/*    plugins/*
compat.scm           kernel/logic/*     utils/regexp/*     languages/*
abbrevs.scm          kernel/texmacs/*   utils/logic/*      ...
debug.scm
srfi.scm
```

### Exit Criteria Per Sub-phase

- **C1**: All files under `progs/kernel/` load without uncaught errors
- **C2**: All files under `progs/utils/` load without uncaught errors
- **C3**: Remaining `progs/` directories (generic, plugins, languages, etc.) load

### Testing Strategy

Each sub-phase gets a lit test (e.g., `test/module/texmacs/load_kernel_library.scm`) with a companion `.script` file checking for a success marker.

## Phase Ordering

```
A1 -> A2 -> A3    (sequential, criticality order)
B                  (independent, any time)
C1 -> C2 -> C3    (sequential, depends on A track complete)
```

## Risks

- **A2 GC error throwing**: The GC may be called in contexts where throwing a Scheme error is awkward (e.g., inside a C function). Mitigation: the error is non-recoverable by design — it signals to the embedding app that memory is exhausted.
- **C track unknowns**: Real TeXmacs files may reveal semantic differences that require deeper pscm changes (macro hygiene, module resolution order). Mitigation: stop and write a separate design spec if any single issue requires >2 files changed.
- **C++ stub realism**: C track will populate many C++ stubs. These are sufficient for "loads without errors" but a real TeXmacs build would need actual implementations. This is a separate follow-on project.

## Non-goals

- Multi-threading support
- Performance parity with Guile 1.8
- hygienic macros (syntax-rules)
- Full TeXmacs GUI integration (real C++ function implementations)
- GOOPS
