# Kernel Loading Gap Fixes

**Date**: 2026-05-05
**Goal**: Fix 4 remaining issues blocking kernel layer loading: `stack` not found, `texmacs-module` not found, `'|'` parser gap, `kernel-base` scope.

## Issues and Fixes

### 1. `stack` not found (init-texmacs.scm:17)

**Root cause**: `(debug-set! stack 2000000)` — `stack` is evaluated as a variable but is not defined. In Guile 1.8, `debug-set!` is a special form that does not evaluate its first argument. pscm's stub `(define (debug-set! key val) (noop))` is a regular function that evaluates all arguments.

**Fix**: Change the stub from `define` to `define-macro`:
```scheme
(define-macro (debug-set! key val) '(noop))
```
This matches Guile 1.8 semantics — arguments are not evaluated before the call.

### 2. `texmacs-module` not found in remaining kernel files

**Root cause**: init-texmacs.scm fails before loading boot.scm (due to issue 1), so `texmacs-module` is never defined. Boot.scm defines `texmacs-module` via `define-macro`, which (with the module-aware fix in commit 5c0280c) goes into the module obarray.

**Fix**: Issue 1 fix allows init-texmacs.scm to load completely → boot.scm loads → `texmacs-module` defined → visible to subsequently loaded files via g_env macro fallback (commit ae8ea2a).

### 3. `'|'` character literal parse error

**Root cause**: `menu-convert.scm:953` uses `'|` — the character literal for the pipe symbol. pscm's parser does not recognize `|` as a valid character literal. This affects `menu-convert.scm` and `menu-widget.scm` in the gui kernel files.

**Fix**: Modify `src/c/parse.cc` character literal parsing to accept `|` (0x7c). The fix should allow any printable ASCII punctuation character in `#\` character literals, consistent with R5RS character literal syntax.

### 4. `kernel-base` scope in test harness

**Root cause**: `load_kernel.scm` attempts `(set-current-module #f)` which errors (both in pscm and Guile 1.8 — `#f` is not a module). The `--test` mode catches the error and continues, so stubs are actually defined in `(pscm-user)`. However, after init-texmacs.scm loads and creates new modules, the module context may shift, making `kernel-base` inaccessible.

**Fix**: Remove the `(set-current-module #f)` / save / restore dance. Define all stubs and `kernel-base` directly in the current module. Ensure `kernel-base` and other try-load dependencies are defined in a scope visible to the `for-each` loop.

## Scope

These 4 fixes are part of Phase 1 (Kernel Layer) of the production readiness roadmap. They unblock the remaining kernel files (gui, old-gui, logic tests) that were previously inaccessible due to the init chain break.

## Non-goals

- Full debug system implementation (stub is sufficient for init)
- Complete parser character literal audit (fix only `|` and any other missing punctuation)
- Production-ready `set-current-module #f` support (Guile 1.8 doesn't support it either)
