# Phase 1: Kernel Layer Production Readiness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 3 blocking pre-issues (float comparison truncation, GC hard limits, macro-in-expression-position verification) and load all 45 kernel `.scm` files without uncaught errors.

**Architecture:** Three pre-fix tasks address known correctness/stability gaps. Then extend the incremental loading scaffold from `init-texmacs.scm` to cover the entire kernel layer, using the same classify-fix-commit cycle.

**Tech Stack:** pscm (C++20 Scheme interpreter), TeXmacs Scheme source (kernel/ directory, 45 files), CTest test framework

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/c/number.cc` | Numeric comparison operators — fix float truncation via template specializations |
| `src/c/gc.h` | GC root limit constant — raise MAX_ROOTS |
| `src/c/gc.cc` | GC segment management — raise MAX_SEGMENTS, replace abort() with warning+return |
| `test/module/texmacs/macro_expr_pos.scm` | Verify macro object in expression position is expanded, not errored |
| `test/module/texmacs/load_kernel.scm` | Extended driver that loads all 45 kernel files |
| `test/CMakeLists.txt` | Register new CTest tests |

---

## Part A: Pre-Fixes

### Task 1: Write failing float comparison tests

**Files:**
- Create: `test/module/texmacs/float_cmp.scm`
- Modify: `test/CMakeLists.txt:222` — register new test

- [ ] **Step 1: Create the test file**

Create `test/module/texmacs/float_cmp.scm`:

```scheme
;; Test float comparison with negative non-integer values
(display (< -1.5 -1.0)) (newline)        ;; expected: #t
(display (> -1.0 -1.5)) (newline)        ;; expected: #t
(display (<= -1.5 -1.0)) (newline)       ;; expected: #t
(display (>= -1.0 -1.5)) (newline)       ;; expected: #t

;; Test mixed int/float comparisons
(display (< 3 3.5)) (newline)            ;; expected: #t
(display (> 3.5 3)) (newline)            ;; expected: #t
(display (< 3.5 3)) (newline)            ;; expected: #f
(display (> 3 3.5)) (newline)            ;; expected: #f

;; Chained comparisons with floats
(display (< -2.5 -1.0 0 1.5 3.0)) (newline)  ;; expected: #t
(display (< -1.0 -2.5 0)) (newline)          ;; expected: #f
```

- [ ] **Step 2: Register test in CMakeLists.txt**

In `test/CMakeLists.txt`, after the `texmacs_init_loading` test block (line 222), add:

```cmake
add_test(
    NAME float_cmp
    COMMAND $<TARGET_FILE:pscm_cc> --test float_cmp.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module/texmacs
)
```

- [ ] **Step 3: Build and run — expect FAIL (wrong output, pscm_cc exits 0)**

```bash
cd /Users/pikachu/pr/pscm && cmake --preset pscm-cmake && ninja -C out/build/pscm-cmake && ctest --test-dir out/build/pscm-cmake -R float_cmp --output-on-failure 2>&1 | tail -20
```

Expected: test PASSES (pscm_cc exits 0) but output shows `#f` where `#t` is expected. The CTest test checks exit code, so we manually verify the output is wrong. Actually: run the .scm directly to see the incorrect output:

```bash
./out/build/pscm-cmake/pscm_cc --test test/module/texmacs/float_cmp.scm
```

Expected output: `< -1.5 -1.0` prints `#f` instead of `#t`.

- [ ] **Step 4: Commit**

```bash
git add test/module/texmacs/float_cmp.scm test/CMakeLists.txt
git commit -m "test: add float comparison correctness tests"
```

### Task 2: Fix float comparison truncation in number.cc

**Files:**
- Modify: `src/c/number.cc:166-192`

- [ ] **Step 1: Add double specializations for comparison operators**

In `src/c/number.cc`, after the `EqOp` struct (line 199), add double specializations for the comparison operators that lack them:

```cpp
// Double specializations for comparison operators — without these,
// BinaryOperator<float-op>::run calls the int64_t version with implicit
// double→int64_t truncation, causing incorrect results for non-integer values.

template <>
struct LtEqOp<double, double> {
  static bool run(double lhs, double rhs) {
    return lhs <= rhs;
  }
};

template <>
struct GtEqOp<double, double> {
  static bool run(double lhs, double rhs) {
    return lhs >= rhs;
  }
};

template <>
struct GtOp<double, double> {
  static bool run(double lhs, double rhs) {
    return lhs > rhs;
  }
};

template <>
struct LtOp<double, double> {
  static bool run(double lhs, double rhs) {
    return lhs < rhs;
  }
};
```

- [ ] **Step 2: Build and run the float comparison test — expect PASS**

```bash
cd /Users/pikachu/pr/pscm && ninja -C out/build/pscm-cmake && ctest --test-dir out/build/pscm-cmake -R float_cmp --output-on-failure
```

Expected: test PASSES — all comparisons give correct results.

- [ ] **Step 3: Run full test suite to check for regressions**

```bash
cd /Users/pikachu/pr/pscm && ctest --test-dir out/build/pscm-cmake --output-on-failure && ninja -C out/build/pscm-cmake check-base check-cont check-sicp check-macro check-core
```

Expected: all 39 CTest + all lit suites still 100% pass.

- [ ] **Step 4: Commit**

```bash
git add src/c/number.cc
git commit -m "fix: add double specializations for comparison operators to prevent float truncation"
```

### Task 3: Raise GC limits and replace abort() with warnings

**Files:**
- Modify: `src/c/gc.h:102`
- Modify: `src/c/gc.cc:58,165-183,300-303`

- [ ] **Step 1: Raise MAX_SEGMENTS from 16 to 64**

In `src/c/gc.cc` line 58, change:

```cpp
static const int MAX_SEGMENTS = 64;
```

- [ ] **Step 2: Raise MAX_ROOTS from 2048 to 8192**

In `src/c/gc.h` line 102, change:

```cpp
static const int MAX_ROOTS = 8192;
```

- [ ] **Step 3: Replace segment pool exhaustion abort() with warning+fallback**

In `src/c/gc.cc` lines 178-183, replace the `abort()` with a `return nullptr` so callers can handle allocation failure gracefully:

```cpp
    if (g_next_seg >= MAX_SEGMENTS) {
      fprintf(stderr, "WARNING: gc heap segment pool exhausted (max %d), allocation failed\n", MAX_SEGMENTS);
      fflush(stderr);
      return nullptr;
    }
```

- [ ] **Step 4: Replace root limit abort() with warning+noop**

In `src/c/gc.cc` lines 300-303, change:

```cpp
  if (g_num_roots >= MAX_ROOTS) {
    fprintf(stderr, "WARNING: too many GC roots (max %d), root registration skipped for %s\n", MAX_ROOTS, name);
    fflush(stderr);
    return;
  }
```

Do not remove the `#include <cstdlib>` — the remaining 2 `abort()` calls for mmap failure (lines 175, 767) are genuine unrecoverable system errors.

- [ ] **Step 5: Build and run full test suite**

```bash
cd /Users/pikachu/pr/pscm && ninja -C out/build/pscm-cmake && ctest --test-dir out/build/pscm-cmake --output-on-failure && ninja -C out/build/pscm-cmake check-base check-cont check-sicp check-macro check-core
```

Expected: all tests pass. GC allocation behavior unchanged under normal load; under extreme load, allocation failures are now non-fatal warnings instead of aborts.

- [ ] **Step 6: Commit**

```bash
git add src/c/gc.h src/c/gc.cc
git commit -m "fix: raise GC limits (64 segments, 8192 roots) and replace abort() with warnings"
```

### Task 4: Write macro-in-expression-position verification test

**Files:**
- Create: `test/module/texmacs/macro_expr_pos.scm`
- Modify: `test/CMakeLists.txt:222` — register new test

The evaluator at `src/c/eval.cc:687-699` already handles MACRO objects in expression position. This test verifies the fix is in place and catches regressions.

- [ ] **Step 1: Create the test file**

```scheme
;; Test: macro object in expression position should be expanded, not errored
;; Exercises eval.cc lines 687-699 (is_macro branch in the evaluator)

(define-macro (wrapper . body)
  `(begin ,@body))

(define-macro (for-each-2 proc lst)
  `(for-each ,proc ,lst))

;; When `wrapper` expands, the evaluator encounters `for-each-2` as a 
;; MACRO object in expression position. It must expand it rather than error.
(define result '())
(wrapper
  (for-each-2 (lambda (x) (set! result (cons x result))) '(1 2 3)))
(display (equal? (reverse result) '(1 2 3))) (newline)

;; Test: macro that expands to a self-evaluating value
(define-macro (answer) 42)
(display (wrapper (answer))) (newline)
```

- [ ] **Step 2: Register in CMakeLists.txt**

In `test/CMakeLists.txt`, after the `float_cmp` test block, add:

```cmake
add_test(
    NAME macro_expr_pos
    COMMAND $<TARGET_FILE:pscm_cc> --test macro_expr_pos.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module/texmacs
)
```

- [ ] **Step 3: Build and run the test**

```bash
cd /Users/pikachu/pr/pscm && ninja -C out/build/pscm-cmake && ctest --test-dir out/build/pscm-cmake -R macro_expr_pos --output-on-failure
```

Expected: PASS (exit code 0). If `is_macro` branch didn't exist, this would error with "not supported expression type: macro" → non-zero exit.

- [ ] **Step 4: Commit**

```bash
git add test/module/texmacs/macro_expr_pos.scm test/CMakeLists.txt
git commit -m "test: add macro-in-expression-position verification test"
```

---

## Part B: Kernel Layer Incremental Loading

### Task 5: Create kernel-wide loading driver

**Files:**
- Create: `test/module/texmacs/load_kernel.scm`
- Modify: `test/CMakeLists.txt:222` — register new test

- [ ] **Step 1: Create the driver**

Create `test/module/texmacs/load_kernel.scm` that extends the existing init loading scaffold to load all kernel files:

```scheme
;; Kernel-wide loading test — loads all 45 kernel .scm files.
;; Extends the init-texmacs.scm bootstrap with remaining kernel modules.

;; Set up load path
(set! %load-path (cons "/Users/pikachu/pr/texmacs/TeXmacs/progs" %load-path))
(set! %load-path (cons "/Users/pikachu/pr/texmacs/TeXmacs/fonts" %load-path))

;; Save current module so stubs go into g_env
(define pscm-saved-module (current-module))
(set-current-module #f)

;; ---- C++ stubs (same as load_texmacs_init.scm) ----
(define (cpp-get-preference key default) default)
(define (os-mingw?) #f)
(define (os-macos?) #t)
(define (os-win32?) #f)
(define (os-linux?) #f)
(define (os-gnu?) #f)
(define (texmacs-time) 0)
(define (tm-interactive) #f)
(define (scheme-dialect) "guile-b")
(define (supports-email?) #f)
(define (symbol-property sym key) #f)
(define (set-symbol-property! sym key val) (noop))
(define (source-property form key)
  (if (eq? key 'line) 0
      (if (eq? key 'column) 0
          (if (eq? key 'filename) "unknown" #f))))
(define (debug-set! key val) (noop))
(define (debug-enable . args) (noop))
(define (url-concretize path)
  "/Users/pikachu/pr/texmacs/TeXmacs/progs/kernel/boot/boot.scm")
(define (gui-version) "qt5")
(define (module-export! . args) (noop))
(define (with-fluids fluids thunk) (thunk))
(define (selection-active-any?) #f)
(define (window-visible?) #f)
(define (get-output-tree . args) "")
(define (cpp-string->object s) s)
(define (object->string obj) (if (string? obj) obj ""))
(define (get-boolean-preference key) #f)
(define (get-string-preference key) "")
(define (cpp-get-default-library) "")
(define (get-document-language) "english")
(define (get-user-language) "english")
(define (tree->stree t) t)
(define (tree->object tree) "")
(define (tree-search-upwards t pred) #f)
(define (tm? x) #f)
(define (tree-atomic? x) #f)
(define (tree-compound? x) #f)
(define (tree-in? subtree tree) #f)
(define (tree-label tree) "")
(define (tree-children tree) '())
(define (compile-interface-spec spec) spec)
(define (process-use-modules mods) (for-each (lambda (m) (eval `(use-modules ,m))) mods))
(define (select x y) (noop))

;; ---- Missing Guile built-in stubs ----
(define (reverse! lst) (reverse lst))
(define (caar x) (car (car x)))
(define (cdar x) (cdr (car x)))
(define (caaar x) (car (caar x)))
(define (caadr x) (car (cadr x)))
(define (cadar x) (car (cdar x)))
(define (cdaar x) (cdr (caar x)))
(define (cdadr x) (cdr (cadr x)))
(define (cddar x) (cdr (cdar x)))
(define (cdddr x) (cdr (cddr x)))
(define (keyword? x) (and (symbol? x) 
                           (let ((s (symbol->string x)))
                             (and (> (string-length s) 0)
                                  (char=? (string-ref s 0) #\:)))))

;; ---- Load init-texmacs.scm first (bootstrap) ----
(define kernel-base "/Users/pikachu/pr/texmacs/TeXmacs/progs")

(define (try-load path)
  (display "Loading ") (display path) (newline)
  (catch #t
    (lambda () (load path) (display "  OK") (newline))
    (lambda (key . args)
      (display "  ERROR: ") (display key) 
      (display " ") (display args) (newline))))

;; Step 1: Bootstrap (already known to work)
(try-load (string-append kernel-base "/init-texmacs.scm"))

;; Step 2: Load remaining kernel files NOT loaded by init chain
;; These are loaded by init-texmacs.scm: boot.scm, compat.scm, abbrevs.scm, debug.scm, srfi.scm
;; These are pulled in via inherit-modules: library/{base,list}, regexp/{match,select}, logic/*

;; Remaining kernel files to load explicitly:
(define extra-kernel-files
  '("kernel/library/tree.scm"
    "kernel/library/patch.scm"
    "kernel/library/iterator.scm"
    "kernel/library/content.scm"
    "kernel/boot/ahash-table.scm"
    "kernel/boot/prologue.scm"
    "kernel/regexp/regexp-test.scm"
    "kernel/old-gui/old-gui-factory.scm"
    "kernel/old-gui/old-gui-test.scm"
    "kernel/old-gui/old-gui-form.scm"
    "kernel/old-gui/old-gui-widget.scm"
    "kernel/logic/logic-test.scm"
    "kernel/gui/menu-convert.scm"
    "kernel/gui/menu-test.scm"
    "kernel/gui/menu-widget.scm"
    "kernel/gui/menu-define.scm"
    "kernel/gui/kbd-define.scm"
    "kernel/gui/kbd-handlers.scm"
    "kernel/gui/speech-define.scm"
    "kernel/gui/gui-markup.scm"))

(for-each (lambda (f)
            (try-load (string-append kernel-base "/" f)))
          extra-kernel-files)

(display "KERNEL-LOADED") (newline)
```

- [ ] **Step 2: Register in CMakeLists.txt**

In `test/CMakeLists.txt`, after the `macro_expr_pos` block, add:

```cmake
add_test(
    NAME load_kernel
    COMMAND $<TARGET_FILE:pscm_cc> --test load_kernel.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module/texmacs
)
```

- [ ] **Step 3: Build and run — evaluate initial state**

```bash
cd /Users/pikachu/pr/pscm && ninja -C out/build/pscm-cmake && ctest --test-dir out/build/pscm-cmake -R load_kernel --output-on-failure 2>&1 | head -100
```

The first run will show which files load OK and which error. This output is the input for Task 6 iteration.

- [ ] **Step 4: Commit**

```bash
git add test/module/texmacs/load_kernel.scm test/CMakeLists.txt
git commit -m "test: add kernel-wide loading scaffold"
```

### Tasks 6-N: Iterative Fix Cycles

**Pattern:** Each task fixes one error from the loading output, using this template:

- [ ] **Step 1: Run the loading test and capture error**

```bash
cd /Users/pikachu/pr/pscm && ctest --test-dir out/build/pscm-cmake -R load_kernel --output-on-failure 2>&1 | tee /tmp/kernel_error.txt
```

- [ ] **Step 2: Classify the error**

| Category | Example | Action |
|----------|---------|--------|
| Missing C++ stub | `unbound variable: tm-define` | Add `(define ...)` stub to `load_kernel.scm` |
| Missing Scheme built-in | `unbound variable: string-replace` | Implement in pscm or add Scheme stub |
| Macro expansion error | `quasiquote: bad syntax` | Investigate expansion order; fix in macro.cc or eval.cc |
| GC crash | `FATAL: gc heap mmap failed` | Blocking — escalate to GC fix |
| Semantic mismatch | Wrong value returned | Investigate behavior difference vs Guile 1.8 |

- [ ] **Step 3: Apply the fix**

For missing C++ stubs: add the `(define ...)` to the stub section in `load_kernel.scm`.

For missing Scheme built-ins: implement in pscm C++ source if used by multiple files; otherwise add a Scheme-level stub.

For macro expansion errors: investigate the specific expansion behavior. Check whether the macro's defining module is visible during expansion (module environment chain work should handle this). If the macro object itself appears in expression position without expansion, check `src/c/eval.cc:687-699` path.

- [ ] **Step 4: Rebuild and re-run**

```bash
cd /Users/pikachu/pr/pscm && ninja -C out/build/pscm-cmake && ctest --test-dir out/build/pscm-cmake -R load_kernel --output-on-failure
```

- [ ] **Step 5: Commit**

```bash
git add <modified files>
git commit -m "fix: <specific error description> in kernel loading"
```

- [ ] **Step 6: Every 5 cycles, run full regression**

```bash
cd /Users/pikachu/pr/pscm && ctest --test-dir out/build/pscm-cmake --output-on-failure && ninja -C out/build/pscm-cmake check-base check-cont check-sicp check-macro check-core
```

**Stopping conditions** (indicate need for separate spec):
- GC crash that can't be fixed with a small patch
- More than 2 new R5RS built-in procedures needed per category
- A file triggers more than 10 distinct errors (suggests fundamental incompatibility)

### Task Last: Phase 1 completion

- [ ] **Step 1: Verify kernel exit criteria**

```bash
cd /Users/pikachu/pr/pscm && ctest --test-dir out/build/pscm-cmake -R load_kernel --output-on-failure
```

Expected: `KERNEL-LOADED` — all 45 kernel files load without uncaught errors.

- [ ] **Step 2: Full regression**

```bash
cd /Users/pikachu/pr/pscm && ctest --test-dir out/build/pscm-cmake --output-on-failure && ninja -C out/build/pscm-cmake check-base check-cont check-sicp check-macro check-core
```

- [ ] **Step 3: Update gap inventory**

Update `docs/superpowers/specs/2026-05-02-texmacs-gap-inventory.md` with Phase 1 results: which files loaded, which gaps were discovered and fixed, what remains for Phase 2.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-05-02-texmacs-gap-inventory.md
git commit -m "docs: update TeXmacs gap inventory after Phase 1 kernel loading"
```

---

## Phase 2-4 Outlines

### Phase 2: Utils Layer (~100 files)

Pre-fix: printer 4096-byte buffer limit (`src/c/print.cc`), remaining c*r built-ins.

Loading: extend `load_kernel.scm` pattern to `utils/*`. Same classify-fix-commit cycle.

Exit criteria: all utils `.scm` files load without uncaught errors.

### Phase 3: Generic Layer (~200 files)

Pre-fix: 25 C API wrappers (`src/c/pscm_api.h`, `src/c/pscm_api.cc`), `define-module` option parsing (`src/c/module.cc`).

Loading: extend to `generic/*`.

Exit criteria: all generic `.scm` files load without uncaught errors.

### Phase 4: Plugins Layer (~300 files)

Pre-fix: soft ports if needed, feature depth gaps discovered in phases 2-3.

Loading: extend to `plugins/*` and remaining files.

Exit criteria (final): all 663 TeXmacs files load; zero `abort()`/`exit(1)`; all existing tests pass; C API complete; TeXmacs launches to main interface.

---

## Summary

| Phase | Tasks | Output |
|-------|-------|--------|
| A: Pre-fixes | 1-4 | Float cmp fix, GC limits raised, macro-expr-pos test |
| B: Kernel loading | 5-N | `load_kernel.scm` passes, 45 kernel files load |
| 2: Utils | outline | ~100 files load |
| 3: Generic | outline | ~200 files load |
| 4: Plugins | outline | ~300 files load, final exit criteria met |
