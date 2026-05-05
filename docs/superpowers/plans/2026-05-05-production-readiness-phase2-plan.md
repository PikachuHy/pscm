# Production Readiness Phase 2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix remaining stability issue (print buffer truncation), complete C API Phase B module wrappers, and incrementally load all TeXmacs kernel/ modules through pscm.

**Architecture:** Three tracks. Track A3 is the sole remaining stability fix (A1 and A2 were already completed post-audit). Track B is 6 thin C API wrappers. Track C feeds TeXmacs kernel/ modules to pscm in an iterative find-and-fix loop. Tracks A3 and B are independent. Track C can start in parallel with A3/B since init-texmacs.scm already loads.

**Tech Stack:** pscm (C++20 Scheme interpreter), TeXmacs Scheme source (~442 files under progs/), lit test framework

---
## Pre-existing State

A1 (float comparison) and A2 (GC abort()) were already fixed in commits after the 2026-05-02 audit:
- `63b94ad` / `49d3cf1`: float comparison double promotion + dead template cleanup
- `e9bd053`: GC limits raised (MAX_SEGMENTS 16→64, MAX_ROOTS 2048→8192), runtime abort() replaced with warnings
- `e07875b`: public C API wrapped in scm_c_catch to prevent exit(1) on Scheme errors

Only A3 (print buffer) remains from Track A.

---

## Track A3: Fix Print Buffer 4096-Byte Truncation

### Task 1: Fix Port-Output Buffer in write/display

**Files:**
- Modify: `src/c/string.cc:186-250`

**Problem:** `scm_c_display` and `scm_c_write` use a fixed 4096-byte stack buffer when outputting to a port. Output larger than 4095 bytes is silently truncated.

**Affected code paths:**
- `scm_c_display` (line 188): `char buffer[4096]` + `fread(buffer, 1, sizeof(buffer) - 1, tmp_file)`
- `scm_c_write` (line 235): same pattern

The stdout path (no port argument) is unaffected — it calls `print_ast` directly.

- [ ] **Step 1: Replace fixed buffer with dynamic allocation in scm_c_display**

In `src/c/string.cc`, replace lines 188-203:

```cpp
// Before (lines 188-203):
    char buffer[4096];
    FILE *old_stdout = stdout;
    FILE *tmp_file = tmpfile();
    if (!tmp_file) {
      print_ast(obj, false);
      return scm_none();
    }
    stdout = tmp_file;
    print_ast(obj, false);
    fflush(tmp_file);
    rewind(tmp_file);
    size_t len = fread(buffer, 1, sizeof(buffer) - 1, tmp_file);
    buffer[len] = '\0';
    stdout = old_stdout;
    fclose(tmp_file);
    write_string_to_port(port, buffer);
```

Replace with:

```cpp
    FILE *old_stdout = stdout;
    FILE *tmp_file = tmpfile();
    if (!tmp_file) {
      print_ast(obj, false);
      return scm_none();
    }
    stdout = tmp_file;
    print_ast(obj, false);
    fflush(tmp_file);
    long file_size = ftell(tmp_file);
    rewind(tmp_file);
    size_t buf_size = (file_size > 0) ? (size_t)file_size + 1 : 4096;
    char *buffer = (char *)malloc(buf_size);
    if (!buffer) {
      stdout = old_stdout;
      fclose(tmp_file);
      eval_error("display: memory allocation failed");
      return nullptr;
    }
    size_t total = 0;
    while (total < buf_size - 1) {
      size_t n = fread(buffer + total, 1, buf_size - 1 - total, tmp_file);
      if (n == 0) break;
      total += n;
    }
    buffer[total] = '\0';
    stdout = old_stdout;
    fclose(tmp_file);
    write_string_to_port(port, buffer);
    free(buffer);
```

- [ ] **Step 2: Apply same fix to scm_c_write**

Replace lines 236-250 in `src/c/string.cc` with the same dynamic-allocation pattern, using `print_ast(obj, true)` (write mode) instead of `print_ast(obj, false)`.

- [ ] **Step 3: Build and run tests**

```bash
ninja -C out/build/pscm-cmake
ctest --test-dir out/build/pscm-cmake --output-on-failure
```

Expected: all 39 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/c/string.cc
git commit -m "fix: replace fixed 4096-byte buffer with dynamic allocation in display/write port output"
```

---

## Track B: C API Phase B — Module Wrappers

### Task 2: Add pscm_c_resolve_module and pscm_c_module_lookup

**Files:**
- Modify: `src/c/pscm_api.h` (add declarations)
- Modify: `src/c/pscm_api.cc` (add implementations)

- [ ] **Step 1: Add declarations to pscm_api.h**

After the existing `pscm_c_define_gsubr` declaration (line 38), add:

```cpp
// Module operations (compatible with Guile 1.8 API)
SCM *pscm_c_resolve_module(const char *name);                         // Resolve module by name string
SCM *pscm_c_module_lookup(SCM *module, const char *name);             // Look up variable in module
```

- [ ] **Step 2: Add implementations to pscm_api.cc**

After the existing `pscm_c_define_gsubr` implementation, add:

```cpp
SCM *pscm_c_resolve_module(const char *name) {
  return scm_c_resolve_module(name);
}

SCM *pscm_c_module_lookup(SCM *module, const char *name) {
  return scm_c_module_lookup(module, name);
}
```

- [ ] **Step 3: Build**

```bash
ninja -C out/build/pscm-cmake
```

Expected: compiles without errors.

- [ ] **Step 4: Commit**

```bash
git add src/c/pscm_api.h src/c/pscm_api.cc
git commit -m "feat: add pscm_c_resolve_module and pscm_c_module_lookup C API wrappers"
```

### Task 3: Add pscm_c_module_define and pscm_c_use_module

**Files:**
- Modify: `src/c/pscm_api.h`
- Modify: `src/c/pscm_api.cc`

- [ ] **Step 1: Add declarations to pscm_api.h**

After the declarations from Task 2, add:

```cpp
SCM *pscm_c_module_define(SCM *module, const char *name, SCM *val);  // Define variable in module
void  pscm_c_use_module(const char *name);                            // Import module
```

- [ ] **Step 2: Add implementations to pscm_api.cc**

After the implementations from Task 2, add:

```cpp
SCM *pscm_c_module_define(SCM *module, const char *name, SCM *val) {
  return scm_c_module_define(module, name, val);
}

void pscm_c_use_module(const char *name) {
  scm_c_use_module(name);
}
```

- [ ] **Step 3: Build**

```bash
ninja -C out/build/pscm-cmake
```

Expected: compiles without errors.

- [ ] **Step 4: Commit**

```bash
git add src/c/pscm_api.h src/c/pscm_api.cc
git commit -m "feat: add pscm_c_module_define and pscm_c_use_module C API wrappers"
```

### Task 4: Add pscm_c_current_module and pscm_c_set_current_module

**Files:**
- Modify: `src/c/pscm_api.h`
- Modify: `src/c/pscm_api.cc`

- [ ] **Step 1: Add declarations to pscm_api.h**

After the declarations from Task 3, add:

```cpp
SCM *pscm_c_current_module(void);                                     // Get current module
SCM *pscm_c_set_current_module(SCM *module);                          // Set current module
```

- [ ] **Step 2: Add implementations to pscm_api.cc**

```cpp
SCM *pscm_c_current_module(void) {
  return scm_current_module();
}

SCM *pscm_c_set_current_module(SCM *module) {
  return scm_set_current_module(module);
}
```

- [ ] **Step 3: Build and run tests**

```bash
ninja -C out/build/pscm-cmake
ctest --test-dir out/build/pscm-cmake --output-on-failure
```

Expected: all 39 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/c/pscm_api.h src/c/pscm_api.cc
git commit -m "feat: add pscm_c_current_module and pscm_c_set_current_module C API wrappers"
```

---

## Track C: Incremental TeXmacs Module Loading

### Task 5: Create Kernel-Wide Loading Test Driver

**Files:**
- Create: `test/module/texmacs/load_kernel.scm`
- Create: `test/module/texmacs/Output/load_kernel.script`
- Modify: `test/module/CMakeLists.txt`

The existing `load_texmacs_init.scm` loads init-texmacs.scm and the boot chain. This new test extends to load all kernel/ modules. It should load the same stubs + boot chain, then attempt to load files under `progs/kernel/library/`, `progs/kernel/regexp/`, `progs/kernel/logic/`, and `progs/kernel/texmacs/`.

- [ ] **Step 1: Create load_kernel.scm**

```scheme
;; Load all kernel modules through pscm.
;; Extends init-texmacs.scm loading to cover progs/kernel/ subdirectories.

(set! %load-path (cons "/Users/pikachu/pr/texmacs/TeXmacs/progs" %load-path))
(set! %load-path (cons "/Users/pikachu/pr/texmacs/TeXmacs/fonts" %load-path))

;; Save current module so stubs go into g_env (global priority for macro expansion)
(define pscm-saved-module (current-module))
(set-current-module #f)

;; --- C++/TeXmacs stubs (same as load_texmacs_init.scm) ---
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
(define (tree->object t) "")
(define (select . args) (noop))
(define (tree-search-upwards . args) #f)
(define (tm? x) #f)
(define (tree-atomic? x) #f)
(define (tree-compound? x) #f)
(define (tree-in? x y) #f)
(define (tree-label t) "")
(define (tree-children t) '())
(define (compile-interface-spec spec) spec)
(define (process-use-modules mods) (for-each (lambda (m) (eval `(use-modules ,m))) mods))

;; Missing Guile 1.8 built-ins (from audit)
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
(define (keyword? x) #f)
(define (string-replace s old new) s)
(define (string-append-suffix s suffix) (string-append s suffix))
(define (string-append-prefix prefix s) (string-append prefix s))
(define make-ahash-table make-hash-table)
(define ahash-ref hash-ref)
(define ahash-set! hash-set!)
(define ahash-remove! hash-remove!)

;; --- Load boot chain (already known to work) ---
(load "/Users/pikachu/pr/texmacs/TeXmacs/progs/init-texmacs.scm")

;; --- Load kernel/library/ modules ---
(define kernel-library-files
  '("base.scm" "content.scm" "iterator.scm" "list.scm" "patch.scm" "tree.scm"))

(for-each (lambda (f)
  (display "Loading kernel/library/") (display f) (display "...") (newline)
  (load (string-append "/Users/pikachu/pr/texmacs/TeXmacs/progs/kernel/library/" f)))
  kernel-library-files)

;; --- Load kernel/regexp/ modules (if they exist) ---
(define kernel-regexp-files
  '("match.scm" "select.scm"))

(for-each (lambda (f)
  (display "Loading kernel/regexp/") (display f) (display "...") (newline)
  (load (string-append "/Users/pikachu/pr/texmacs/TeXmacs/progs/kernel/regexp/" f)))
  kernel-regexp-files)

;; --- Load kernel/logic/ modules (if they exist) ---
(define kernel-logic-files
  '("rules.scm" "query.scm" "data.scm"))

(for-each (lambda (f)
  (display "Loading kernel/logic/") (display f) (display "...") (newline)
  (load (string-append "/Users/pikachu/pr/texmacs/TeXmacs/progs/kernel/logic/" f)))
  kernel-logic-files)

;; --- Load kernel/texmacs/ modules ---
(define kernel-texmacs-files
  '("tm-define.scm" "tm-preferences.scm" "tm-modes.scm" "tm-language.scm"
    "tm-file-system.scm" "tm-convert.scm" "tm-dialogue.scm"
    "tm-plugins.scm" "tm-secure.scm" "tm-states.scm"))

(for-each (lambda (f)
  (display "Loading kernel/texmacs/") (display f) (display "...") (newline)
  (load (string-append "/Users/pikachu/pr/texmacs/TeXmacs/progs/kernel/texmacs/" f)))
  kernel-texmacs-files)

(display "KERNEL-LOADED") (newline)
```

- [ ] **Step 2: Create Output/load_kernel.script**

```
RUN: %pscm_cc --test %S/load_kernel.scm 2>&1 | %filter
CHECK: KERNEL-LOADED
```

- [ ] **Step 3: Register the test in CMakeLists.txt**

In `test/module/CMakeLists.txt`, after the `texmacs_init_loading` test registration, add:

```cmake
add_test(NAME texmacs_kernel_loading
  COMMAND ${CMAKE_BINARY_DIR}/pscm_cc --test
    ${CMAKE_CURRENT_SOURCE_DIR}/texmacs/load_kernel.scm)
set_tests_properties(texmacs_kernel_loading PROPERTIES
  PASS_REGULAR_EXPRESSION "KERNEL-LOADED")
```

- [ ] **Step 4: Build and run (expect first failure)**

```bash
ninja -C out/build/pscm-cmake
ctest --test-dir out/build/pscm-cmake -R texmacs_kernel_loading --output-on-failure
```

Expected: likely FAIL on first run. The error output is the starting point for the fix cycle below.

- [ ] **Step 5: Commit**

```bash
git add test/module/texmacs/load_kernel.scm test/module/texmacs/Output/load_kernel.script
git add test/module/CMakeLists.txt
git commit -m "test: add kernel-wide TeXmacs module loading test"
```

### Task 6-N: Iterative Fix Cycle for Kernel Modules

**Pattern:** Each subsequent task repeats the cycle: run the kernel loading test → classify the first error → fix it → commit → re-run until KERNEL-LOADED passes.

**Error classification and actions:**

| Category | Example | Action |
|----------|---------|--------|
| Missing Scheme built-in | `unbound variable: read-line` | Add stub in load_kernel.scm for now, file issue for real implementation |
| Missing C++ stub | `unbound variable: tree-search-upwards` | Add `(define (tree-search-upwards . args) #f)` to stub section |
| Syntax/macro mismatch | `quasiquote: bad syntax` | Analyze vs Guile 1.8, fix in pscm source |
| Module resolution failure | `no code for module (kernel ...)` | Check load path, module naming |
| Macro-in-expression-position | `not supported expression type: macro` | Fix in pscm evaluator |

**Per-iteration commit template:**

```bash
git add <modified files>
git commit -m "fix: <specific error description> in kernel module loading"
```

**Expected iterations:** 5-15 cycles depending on gap severity. Early failures most likely from missing C++ stubs. Later failures from syntax/macro differences.

**Stopping conditions** (require a separate design spec):
- A single issue requires changes to >3 pscm source files
- A macro hygiene or module resolution difference that requires architectural change
- GC crash that can't be fixed by raising limits further

### Task Last: Kernel Loading Milestone

- [ ] **Step 1: Verify KERNEL-LOADED**

```bash
ctest --test-dir out/build/pscm-cmake -R texmacs_kernel_loading --output-on-failure
```

Expected: PASS with `KERNEL-LOADED` in output.

- [ ] **Step 2: Run full test suite to confirm no regressions**

```bash
ctest --test-dir out/build/pscm-cmake --output-on-failure
ninja -C out/build/pscm-cmake check-base check-cont check-sicp check-macro check-core
```

Expected: all 39 ctest tests + all lit suites pass.

- [ ] **Step 3: Update gap inventory document**

Update `docs/superpowers/specs/2026-05-02-texmacs-gap-inventory.md`:
- Mark kernel/ modules as loaded
- List any remaining stubs added during the fix cycles
- Note any architecture-level gaps discovered
- Update recommendation section

- [ ] **Step 4: Commit milestone**

```bash
git add docs/superpowers/specs/2026-05-02-texmacs-gap-inventory.md
git commit -m "docs: update TeXmacs gap inventory with kernel loading milestone"
```

---

## Summary

| Task | Track | Output |
|------|-------|--------|
| 1 | A3 | Fix 4096-byte buffer in display/write port output |
| 2 | B | `pscm_c_resolve_module`, `pscm_c_module_lookup` |
| 3 | B | `pscm_c_module_define`, `pscm_c_use_module` |
| 4 | B | `pscm_c_current_module`, `pscm_c_set_current_module` |
| 5 | C | Kernel-wide loading test driver |
| 6-N | C | Iterative fix cycles until KERNEL-LOADED |
| Last | C | Milestone: kernel loading passes, gap inventory updated |

Tracks A3 (Task 1) and B (Tasks 2-4) are independent and can be done in any order. Track C (Tasks 5-N) depends on Task 1 for print correctness but can otherwise start immediately — the kernel loading test will surface real issues regardless.
