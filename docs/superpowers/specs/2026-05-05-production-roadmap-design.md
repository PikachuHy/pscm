# Production Readiness Roadmap: TeXmacs Compatibility

**Date**: 2026-05-05
**Goal**: Make pscm a production-grade Guile 1.8 drop-in replacement for TeXmacs, covering all 663 Scheme files.

## Current Baseline

**Achieved:**
- init-texmacs.scm loads without uncaught errors
- kernel/boot/* fully loaded (boot.scm, compat.scm, abbrevs.scm, debug.scm, srfi.scm)
- 39 CTest + all lit suites 100% passing
- Module environment chain macro expansion fixed (commits up to 26e99be)
- Guile 1.8 feature audit complete: 396 present, 13 partial, 88 missing

**Current blocking issues (from gap inventory):**
- Macro in expression position: `for` inside `define-macro` body fails with "not supported expression type: macro" (eval.cc:710)
- Float comparison truncation: `<`, `>`, `<=`, `>=` truncate double to int64
- GC has 4 `abort()` calls on heap exhaustion (MAX_SEGMENTS=16, MAX_ROOTS=2048)
- 25 C API wrappers missing (module operations, type predicates)
- `define-module` option parsing is "simplified version"

## Strategy: Layered Progression (方案 C)

Four phases, each with pre-fix → incremental load → exit criteria.

### Phase 1: Kernel Layer

**Scope:** `kernel/boot/*`, `kernel/library/*`, `kernel/logic/*`, `kernel/texmacs/*` (~50 files)

**Pre-fix before loading:**

| # | Issue | Rationale | Complexity |
|---|-------|-----------|------------|
| 1 | Macro in expression position | Blocks tm-define.scm `for` macro | Small — fix eval.cc macro dispatch |
| 2 | Float comparison truncation | Numeric code in kernel logic will trigger | Small — fix BinaryOperator template |
| 3 | GC segment/root limits | Memory pressure from 50 files | Medium — make configurable or raise limits |

**Loading strategy:**
- Extend `load_texmacs_init.scm` to cover all kernel files
- Same cycle: load → error → classify → fix → commit → repeat
- Regression run every 5 fix cycles

**Exit criteria:** All kernel `.scm` files load without uncaught errors.

### Phase 2: Utils Layer

**Scope:** `utils/*` (~100 files)

**Expected challenges:** Missing Guile built-ins (c*r compositions, string ops), more macro expansion edge cases, printer buffer limit.

**Pre-fix:** Printer 4096-byte buffer limit, remaining missing c*r built-ins if needed.

**Exit criteria:** All utils `.scm` files load without uncaught errors.

### Phase 3: Generic Layer

**Scope:** `generic/*` (~200 files)

**Expected challenges:** C API wrapper gaps exposed, heavy module interaction, soft ports.

**Pre-fix:** 25 C API wrappers (pscm_c_resolve_module, pscm_c_module_lookup, etc.), `define-module` option parsing.

**Exit criteria:** All generic `.scm` files load without uncaught errors.

### Phase 4: Plugins Layer

**Scope:** `plugins/*` and remaining files (~300 files)

**Expected challenges:** Feature depth (syntax-rules?), performance/memory pressure, edge-case interactions.

**Exit criteria:** All 663 TeXmacs Scheme files load without uncaught errors.

## Non-Functional Requirements

| Category | Issue | Target Phase |
|----------|-------|-------------|
| Stability | Eliminate 4 `abort()` in GC — replace with recoverable errors | 1-2 |
| Stability | Eliminate 2 `exit(1)` calls — replace with Scheme exceptions | 1-2 |
| Correctness | Float comparison truncation bug | 1 |
| Correctness | Printer 4096-byte buffer limit | 2 |
| Compatibility | 25 C API wrappers (pscm_c_*) | 2-3 |
| Compatibility | `define-module` option parsing completeness | 1-2 |

## Error Classification (per cycle)

| Category | Example | Action |
|----------|---------|--------|
| Missing C++ stub | `unbound variable: cpp-get-preference` | Add Scheme stub |
| Missing Scheme built-in | `unbound variable: read-line` | Implement or stub |
| Macro expansion error | `quasiquote: bad syntax` | Investigate, fix in pscm |
| GC crash/abort | `FATAL: gc heap mmap failed` | Blocking — fix in pscm |
| Semantic mismatch | Wrong value returned | Investigate, fix or file |

## Final Exit Criteria

- [ ] All 663 TeXmacs Scheme files load without uncaught errors
- [ ] All existing tests continue 100% passing
- [ ] GC has zero `abort()` hard crashes
- [ ] Zero `exit(1)` calls (all converted to Scheme exceptions)
- [ ] C API covers all functions TeXmacs uses
- [ ] TeXmacs launches to main interface

## Non-Goals

- GOOPS (TeXmacs has its own OOP layer)
- Multi-threading (pscm is single-threaded)
- Performance parity with Guile 1.8
- `syntax-rules` / hygienic macros (unless TeXmacs actually requires them)
- TeXmacs plugins beyond what the bootstrap chain loads
