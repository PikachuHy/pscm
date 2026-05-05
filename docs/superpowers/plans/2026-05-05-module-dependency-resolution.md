# Module Dependency Resolution Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `scm_resolve_module` so that module dependencies (`#:use-module`) are correctly resolved and their exports visible during module body evaluation, eliminating the 251 caught errors in cross-file TeXmacs loading.

**Architecture:** Two fixes: (1) in `eval_define_module`, stop skipping option processing when module already exists in registry — instead use the existing module and continue to process `#:use-module` etc. (2) in `eval_define`, detect module environments (`env->module` is set) and route `define` to the module obarray instead of the env chain.

**Tech Stack:** pscm (C++20 Scheme interpreter), lit test framework

---

## Root Cause

`scm_resolve_module` (`module.cc:297`) pre-creates and registers a module skeleton (lines 392-396) **before** loading the source file. When the file's `(define-module ...)` form is evaluated, `eval_define_module` (`module.cc:724`) finds the module already in the registry and **returns immediately** at line 756, skipping all option processing. Dependencies declared via `#:use-module` are never loaded, so their macros (e.g., `tm-define-macro`) are not visible during body evaluation.

Compounding issue: when a file is loaded via `scm_c_primitive_load_with_env`, the environment has `env->parent = &g_env`. `eval_define` interprets `env->parent` as "nested environment" and routes definitions to the env chain rather than the module obarray, so exported symbols can't be found by other modules.

---

### Task 1: Fix eval_define_module — process options on existing module

**Files:**
- Modify: `src/c/module.cc:751-764`

- [ ] **Step 1: Replace the early-return logic**

Current code at lines 751-764:

```cpp
  // Use equal? comparison for module names (lists), not eq?
  SCM *existing = scm_c_hash_ref_equal(wrap(g_module_registry), wrap(name));
  if (existing && !is_falsy(existing) && is_module(existing)) {
    // Module already exists (e.g., created by scm_resolve_module), use it
    scm_set_current_module(existing);
    return scm_none();
  }

  // Create new module
  SCM *module_scm = scm_make_module(name, 31);

  // Register module
  // Use equal? comparison for module names (lists), not eq?
  scm_c_hash_set_equal(wrap(g_module_registry), wrap(name), module_scm);
```

Replace with:

```cpp
  // Use equal? comparison for module names (lists), not eq?
  SCM *existing = scm_c_hash_ref_equal(wrap(g_module_registry), wrap(name));
  bool mod_exists = (existing && !is_falsy(existing) && is_module(existing));

  SCM *module_scm;
  if (mod_exists) {
    // Module already exists (e.g., created by scm_resolve_module), use it
    module_scm = existing;
  } else {
    // Create new module
    module_scm = scm_make_module(name, 31);
    // Register module
    scm_c_hash_set_equal(wrap(g_module_registry), wrap(name), module_scm);
  }
```

The `module_scm` variable then flows into the existing option-processing loop (lines 766-840) and `scm_set_current_module(module_scm)` (line 843). No other logic changes needed — the existing code already references `module_scm` directly.

- [ ] **Step 2: Build and verify no regressions**

```bash
ninja -C out/build/pscm-cmake
ctest --test-dir out/build/pscm-cmake --output-on-failure
```

Expected: all 44 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/c/module.cc
git commit -m "fix: process define-module options on pre-existing module"
```

---

### Task 2: Fix eval_define — route definitions to module obarray in module envs

**Files:**
- Modify: `src/c/define.cc:25-27`

- [ ] **Step 1: Add module environment check to define routing**

Current code at line 25:

```cpp
    if (env->parent) {
      // We're in a nested environment (lambda/let), create local binding
      scm_env_insert(env, varname, val, /*search_parent=*/false);
```

Replace with:

```cpp
    if (env->parent) {
      // Check if this is a module environment (env->module is set by scm_resolve_module)
      if (env->module && is_module(env->module)) {
        SCM_Module *mod = cast<SCM_Module>(env->module);
        scm_c_hash_set_eq(wrap(mod->obarray), wrap(varname), val);
        if (is_proc(val)) {
          auto proc = cast<SCM_Procedure>(val);
          if (proc->name == nullptr) proc->name = varname;
        }
      } else {
        // We're in a nested environment (lambda/let), create local binding
        scm_env_insert(env, varname, val, /*search_parent=*/false);
        if (is_proc(val)) {
          auto proc = cast<SCM_Procedure>(val);
          if (proc->name == nullptr) proc->name = varname;
        }
      }
```

Note the existing line 15-21 above already sets `proc->name` before this routing decision. The `is_proc` check here ensures the name is set regardless of which path is taken.

Actually, looking at the existing code more carefully — the `is_proc` check at lines 15-21 is already above the `if (env->parent)` block. So when we add the module obarray path, the proc name is already set. We just need to route the binding. Simplified:

```cpp
    if (env->parent) {
      // Check if this is a module environment (env->module from scm_resolve_module)
      if (env->module && is_module(env->module)) {
        SCM_Module *mod = cast<SCM_Module>(env->module);
        scm_c_hash_set_eq(wrap(mod->obarray), wrap(varname), val);
      } else {
        // We're in a nested environment (lambda/let), create local binding
        scm_env_insert(env, varname, val, /*search_parent=*/false);
      }
```

- [ ] **Step 2: Build and test**

```bash
ninja -C out/build/pscm-cmake
ctest --test-dir out/build/pscm-cmake --output-on-failure
```

Expected: all 44 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/c/define.cc
git commit -m "fix: route define to module obarray in module environments"
```

---

### Task 3: Create cross-module dependency loading test

**Files:**
- Create: `test/module/cross_module_a.scm`
- Create: `test/module/cross_module_b.scm`
- Create: `test/module/cross_module_loading.scm`
- Modify: `test/CMakeLists.txt`

- [ ] **Step 1: Create dependency module**

Create `test/module/cross_module_a.scm`:

```scheme
(define-module (cross module a)
  #:export (value-a helper-func make-double))

(define value-a 42)
(define (helper-func x) (+ x 1))
(define (make-double x) (* x 2))
```

- [ ] **Step 2: Create consumer module**

Create `test/module/cross_module_b.scm`:

```scheme
(define-module (cross module b)
  #:use-module (cross module a)
  #:export (value-b b-func))

(define value-b (* value-a 2))
(define (b-func x) (helper-func (make-double x)))
```

- [ ] **Step 3: Create test driver**

Create `test/module/cross_module_loading.scm`:

```scheme
;; Test: cross-file module loading via scm_resolve_module
;; Uses resolve-module which triggers scm_resolve_module -> file load path
(set! %load-path (cons "." %load-path))

;; Load via resolve-module (triggers scm_resolve_module pre-creation + file load)
(define mod-a (resolve-module '(cross module a)))
(define mod-b (resolve-module '(cross module b)))

;; Verify module a exports
(define val-a (module-ref mod-a 'value-a))
(if (= val-a 42)
    (display "PASS: cross_module_a value-a is 42") (newline)
    (begin (display "FAIL: cross_module_a value-a is ") (display val-a) (newline) (exit 1)))

;; Verify module b sees module a's exports via uses chain
(define val-b (module-ref mod-b 'value-b))
(if (= val-b 84)
    (display "PASS: cross_module_b value-b is 84") (newline)
    (begin (display "FAIL: cross_module_b value-b is ") (display val-b) (newline) (exit 1)))

;; Verify function cross-module access
(define fb (module-ref mod-b 'b-func))
(define result (fb 5))
(if (= result 11)  ;; 5*2 + 1 = 11
    (display "PASS: cross_module_b b-func(5) is 11") (newline)
    (begin (display "FAIL: cross_module_b b-func(5) is ") (display result) (newline) (exit 1)))

(display "CROSS-MODULE-PASS") (newline)
```

- [ ] **Step 4: Register test in CMakeLists.txt**

In `test/CMakeLists.txt`, after the existing module tests (near the `load_all` test), add:

```cmake

add_test(
    NAME cross_module_loading
    COMMAND $<TARGET_FILE:pscm_cc> --test cross_module_loading.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module
)
set_tests_properties(cross_module_loading PROPERTIES
  PASS_REGULAR_EXPRESSION "CROSS-MODULE-PASS")
```

- [ ] **Step 5: Build and run the new test**

```bash
cd out/build/pscm-cmake && cmake ../../ && cd ../..
ninja -C out/build/pscm-cmake
ctest --test-dir out/build/pscm-cmake -R cross_module_loading --output-on-failure
```

Expected: PASS with "CROSS-MODULE-PASS".

- [ ] **Step 6: Run full test suite**

```bash
ctest --test-dir out/build/pscm-cmake --output-on-failure
ninja -C out/build/pscm-cmake check-base check-cont check-macro check-core check-sicp
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add test/module/cross_module_a.scm test/module/cross_module_b.scm test/module/cross_module_loading.scm test/CMakeLists.txt
git commit -m "test: add cross-module dependency loading test via scm_resolve_module"
```

---

### Task 4: Verify TeXmacs loading improvement

- [ ] **Step 1: Run load_all test and check error reduction**

```bash
ctest --test-dir out/build/pscm-cmake -R load_all --output-on-failure
```

Expected: load_all still passes (ALL-LOADED). Count errors to confirm reduction.

- [ ] **Step 2: Measure error count**

```bash
/Users/pikachu/pr/pscm/out/build/pscm-cmake/pscm_cc --test test/module/texmacs/load_all.scm > /tmp/load_final.txt 2>&1
echo "ERROR count:"
strings /tmp/load_final.txt | grep -c 'ERROR:'
echo "OK count:"
strings /tmp/load_final.txt | grep -c 'OK$'
```

Expected: error count drops from ~251 to significantly fewer. The remaining errors will be from files needing additional C++ stubs (TeXmacs GUI functions).

- [ ] **Step 3: Commit any follow-up fixes**

If specific missing symbols remain that can be stubbed safely (ones NOT defined in kernel modules), add them to `load_kernel.scm`.

---

### Task 5: Update gap inventory

- [ ] **Step 1: Update the texmacs gap inventory doc**

In `docs/superpowers/specs/2026-05-02-texmacs-gap-inventory.md`, update the loading status and recommendation to reflect that module dependency resolution now works correctly.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-05-02-texmacs-gap-inventory.md
git commit -m "docs: update gap inventory with module dependency resolution fix"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Fix `eval_define_module` — process options on existing module | `module.cc:751-764` |
| 2 | Fix `eval_define` — route to module obarray in module envs | `define.cc:25-27` |
| 3 | Create cross-module loading test | 4 files |
| 4 | Verify TeXmacs loading improvement | `load_all.scm` |
| 5 | Update gap inventory | `texmacs-gap-inventory.md` |

Tasks 1 and 2 are independent. Task 3 depends on both. Tasks 4 and 5 are verification/documentation.
