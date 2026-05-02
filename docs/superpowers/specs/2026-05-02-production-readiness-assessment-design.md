# Production Readiness Assessment for TeXmacs Compatibility

**Date**: 2026-05-02
**Goal**: Determine whether pscm can serve as a Guile 1.8 drop-in replacement for TeXmacs (663 Scheme files, ~238K lines), and close the gap to "yes."

## Current State

pscm v0.5.0: ~19K lines C++, 134 test files, 100% pass across CTest and lit suites. Module system core (public interface, binder, autoload, define-module options) recently completed. Error handling hardened—one intentional `exit(1)` remains in `scm_uncaught_throw`. Single-threaded mark-sweep GC with conservative stack scanning, hard limits: 16 heap segments, 2048 roots.

## Approach

Three phases, executed in order. Each phase produces a concrete artifact.

### Phase 1: Systematic Feature Audit

Compare pscm against Guile 1.8's surface area across these categories:

- Special forms (`define`, `lambda`, `if`, `cond`, `case`, `do`, `let*`, `letrec`, `and`, `or`, `delay`, `quasiquote`, `set!`, `begin`, `let`)
- Module system (`define-module`, `use-modules`, `export`, `re-export`, `module-use!`, `module-ref`, `resolve-module`, `module-map`, `define-public`)
- Macros (`define-macro`, macro expansion ordering, hygiene)
- Built-in procedures (list, string, vector, hash table, port, eval, load, dynamic-wind, call/cc, values/call-with-values, apply, map, for-each)
- C API (`pscm_eval`, `pscm_eval_string`, `pscm_parse`, `scm_c_define`, `scm_call_N`, module C API)
- Ports and I/O (file ports, string ports, soft ports, current-input/output/error-port)
- Numeric tower (fixnums, floats, ratios)
- Reader/printer (shared structure, character literals, string escapes, keyword syntax)

For each feature: mark as **present**, **partial**, or **missing**. The audit reads pscm source directly (not docs), so it's a measurement, not a guess.

**Output**: `docs/superpowers/specs/2026-05-02-guile-1.8-feature-audit.md` — an inventory table.

### Phase 2: Focused Compatibility Tests

Based on audit gaps, write targeted `.scm` tests that exercise Guile 1.8 features TeXmacs uses. Priority order:

1. Module system edge cases — nested modules, circular imports, re-exports with name conflicts, `module-use!` ordering
2. Macro expansion — TeXmacs macros (`texmacs-module`, `provide-public`, `inherit-modules`, `tm-define`) that may rely on Guile-specific expansion behavior
3. Port and I/O — string ports for document processing
4. Continuation + dynamic-wind + module interactions

These go into `test/module/texmacs/` alongside existing smoke tests, each with a `.script` expectation file.

**Output**: New test files under `test/module/texmacs/`.

### Phase 3: Incremental Loading

Feed real TeXmacs Scheme files to pscm, iterating:

1. Start with `init-texmacs.scm` (552 lines, the TeXmacs bootstrap)
2. For each error: classify as missing built-in, macro bug, GC issue, or semantic mismatch
3. Fix blocking errors, file non-blocking ones
4. Proceed through the load chain: `progs/kernel/` → `progs/utils/` → `progs/generic/` → ...
5. Repeat until the full init sequence loads without uncaught errors

**Exit criteria**: `init-texmacs.scm` loads without uncaught errors, and the core kernel modules (`progs/kernel/`) resolve correctly.

## Non-goals

- Full TeXmacs GUI functionality (that's C++ side, not Scheme)
- GOOPS (TeXmacs has its own OOP layer, not Guile's GOOPS)
- Multi-threading (pscm is single-threaded by design)
- Performance parity with Guile 1.8
- Plugins beyond what the bootstrap chain loads

## Risks

- **GC stability**: The conservative stack scan may miss roots or falsely retain objects. Stress from 238K lines of real code will surface these bugs.
- **Macro differences**: `define-macro` is inherently unhygienic; subtle expansion order differences can cause hard-to-debug failures.
- **Scope creep**: 663 files is a lot. We constrain to the bootstrap chain first; plugins and optional modules can follow later.

## Transition

After this assessment reaches exit criteria, the next step is a focused implementation plan for any remaining gap categories discovered during Phase 3.
