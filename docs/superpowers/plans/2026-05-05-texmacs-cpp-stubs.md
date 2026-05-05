# TeXmacs C++ Function Stubs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stubs for remaining TeXmacs C++-side functions to reduce caught errors in load_all.scm, without causing hangs.

**Architecture:** After the module dependency resolution fix (b0d431d), module-defined macros like `menu-bind` and `define-preferences` are now correctly resolved. Remaining errors are from C++ functions that TeXmacs registers via `scm_c_define` — these are genuinely missing and safe to stub as Scheme `define`.

**Tech Stack:** pscm (C++20 Scheme interpreter), Scheme stubs in load_kernel.scm

**Pre-existing state:** Stubs were already edited into `load_kernel.scm` at line 274 but not yet committed.

---

### Task 1: Build and verify stubs don't cause hang

**Files:**
- Modify: `test/module/texmacs/load_kernel.scm` (already edited pre-commit)

The stubs added (lines 276-287):

```scheme
;; TeXmacs C++ function stubs — these are genuinely missing (not module-defined macros)
(define (url-none . args) "")
(define (tree-remove t x) t)
(define (tmfs-title-handler . args) #f)
(define (tm-property . args) #f)
(define (tm-call-back . args) #f)
(define (tm-build-widget . args) #f)
(define (server-define-error-codes . args) #f)
(define (new-author . args) '())
(define (make-record-type name fields) (list 'record-type name))
(define (string->url s) s)
(define (bib-define-style . args) #f)
```

- [ ] **Step 1: Run load_kernel and load_utils to verify no hang**

```bash
ctest --test-dir out/build/pscm-cmake -R 'load_kernel|load_utils|cross_module' --output-on-failure
```

Expected: all 3 tests pass. load_utils should complete (not hang).

- [ ] **Step 2: Run full test suite to check for regressions**

```bash
ctest --test-dir out/build/pscm-cmake --output-on-failure
```

Expected: 45/45 pass (cross_module_loading was added).

- [ ] **Step 3: Measure error reduction**

```bash
/Users/pikachu/pr/pscm/out/build/pscm-cmake/pscm_cc --test /Users/pikachu/pr/pscm/test/module/texmacs/load_all.scm > /tmp/load_with_stubs.txt 2>&1 &
PID=$!
sleep 300
if kill -0 $PID 2>/dev/null; then
  kill $PID 2>/dev/null; wait $PID 2>/dev/null
  echo "Still running at 300s — checking partial results"
  strings /tmp/load_with_stubs.txt | grep -c 'ERROR:'
  strings /tmp/load_with_stubs.txt | grep -c 'OK$'
else
  wait $PID
  echo "Completed"
  strings /tmp/load_with_stubs.txt | grep -c 'ERROR:'
  strings /tmp/load_with_stubs.txt | grep -c 'OK$'
  strings /tmp/load_with_stubs.txt | grep 'ALL-LOADED'
fi
```

Expected: error count lower than before stubs were added, no hang.

- [ ] **Step 4: Commit**

```bash
git add test/module/texmacs/load_kernel.scm
git commit -m "feat: add C++ function stubs for remaining TeXmacs symbols"
```

---

### Task 2: Update gap inventory

- [ ] **Step 1: Update the gap inventory document**

In `docs/superpowers/specs/2026-05-02-texmacs-gap-inventory.md`, note that C++ function stubs were added and error count reduced.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-05-02-texmacs-gap-inventory.md
git commit -m "docs: update gap inventory with C++ stub results"
```

---

## Safety Notes

Unlike the failed C4 attempt (which added global macros that conflicted with module-defined ones), these stubs are for C++ functions that are NOT defined in any TeXmacs Scheme module. They use `define` (not `define-macro`), producing function values that won't trigger recursive macro expansion. The earlier issue with `menu-bind` and `define-preferences` occurred because those ARE Scheme macros defined in kernel modules — adding global `define-macro` stubs conflicted with the module system.
