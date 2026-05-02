# Module System Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the module system to Guile 1.8 compatibility by implementing public interface module creation, binder/autoload support, and define-module option parsing.

**Architecture:** Changes span `src/c/module.cc` (logic), `src/c/pscm_types.h` (struct addition), `src/c/parse.cc` (keyword syntax), and `src/c/gc.cc` (trace the new field). Tests use the existing `pscm_cc --test` pattern, placed in `test/module/` and registered in `test/CMakeLists.txt`. The public interface fallback code already exists at three call sites — creating the module makes it all work.

**Tech Stack:** C++20, GC-managed objects, pscm built-in function registration

---

### Task 1: Add keyword syntax to parser

**Files:**
- Modify: `src/c/parse.cc`

`#:keyword` is standard Guile syntax and must be parseable for define-module options.

- [ ] **Step 1: Add `#:` handler in read_hash**

Find the `#` dispatch in `parse.cc`. After `#\\` (character) and `#(` (vector), add:

```cpp
// #:keyword
if (p->pos[0] == '#' && p->pos[1] == ':') {
  p->pos += 2; // skip #:
  // Read the keyword name as a symbol
  SCM_Symbol *sym = read_symbol(p);
  // Return as a special symbol with ":" prefix for matching
  // We'll match on symbol name "use-module" etc. (the "#:" is stripped)
  return wrap(sym);
}
```

- [ ] **Step 2: Build and verify compilation**

```bash
ninja -C out/build/pscm-cmake
```
Expected: build succeeds

- [ ] **Step 3: Verify keyword parsing works**

```bash
echo "(display #:test-keyword)" | ./out/build/pscm-cmake/pscm_cc
```
Expected: prints `#:test-keyword` (or the symbol), no parse error

- [ ] **Step 4: Commit**

```bash
git add src/c/parse.cc
git commit -m "feat: add #:keyword syntax to parser"
```

---

### Task 2: Add autoload_specs to SCM_Module struct and trace it

**Files:**
- Modify: `src/c/pscm_types.h:124-134`

- [ ] **Step 1: Add autoload_specs field**

```cpp
// In struct SCM_Module, after the exports field:
struct SCM_Module {
  SCM_HashTable *obarray;
  SCM_List *uses;
  SCM_Procedure *binder;
  SCM_Procedure *eval_closure;
  SCM_Procedure *transformer;
  SCM_List *name;
  SCM_Symbol *kind;
  SCM_Module *public_interface;
  SCM_List *exports;
  SCM_List *autoload_specs;  // List of (mod-name . sym-list) for autoload
};
```

- [ ] **Step 2: Initialize autoload_specs in scm_make_module**

In `scm_make_module` (module.cc line 31), add:
```cpp
module->autoload_specs = nullptr;
```

- [ ] **Step 3: Add autoload_specs to GC trace_module**

In `src/c/gc.cc`, in `trace_module`, add after the `trace_ptr(mod->exports, stack)` line:
```cpp
  trace_ptr(mod->autoload_specs,   stack); // SCM_List*
```

- [ ] **Step 4: Build and verify compilation**

```bash
ninja -C out/build/pscm-cmake
```
Expected: build succeeds

- [ ] **Step 5: Commit**

```bash
git add src/c/pscm_types.h src/c/module.cc src/c/gc.cc
git commit -m "feat: add autoload_specs field to SCM_Module with GC tracing"
```

---

### Task 3: Implement public interface module creation

**Files:**
- Modify: `src/c/module.cc:379-397`

- [ ] **Step 1: Write scm_ensure_public_interface function**

Insert before `scm_module_export` (before line 380):

```cpp
// Ensure module has a public interface module, creating one if needed.
// The public interface is a separate module whose obarray contains
// only exported bindings.  Returns the interface module.
static SCM_Module *scm_ensure_public_interface(SCM_Module *module) {
  if (module->public_interface) {
    return module->public_interface;
  }

  // Create the interface module
  SCM_Module *iface = cast<SCM_Module>(scm_make_module(module->name, 31));
  iface->kind = make_sym("interface");
  iface->public_interface = module; // bidirectional

  module->public_interface = iface;

  // Populate interface obarray from exports list
  SCM_List *exp = module->exports;
  while (exp) {
    if (exp->data && is_sym(exp->data)) {
      SCM_Symbol *sym = cast<SCM_Symbol>(exp->data);
      // Copy the binding value from module's obarray
      SCM *val = module_obarray_lookup(module, sym);
      if (val) {
        scm_c_hash_set_eq(wrap(iface->obarray), wrap(sym), val);
      }
    }
    exp = exp->next;
  }

  return iface;
}
```

- [ ] **Step 2: Modify scm_module_export to call scm_ensure_public_interface**

Replace the body of `scm_module_export` (lines 380-390):
```cpp
void scm_module_export(SCM_Module *module, SCM_Symbol *sym) {
  // Add to export list
  SCM_List *new_export = make_list(wrap(sym));
  if (module->exports) {
    new_export->next = module->exports;
  }
  module->exports = new_export;

  // Update public interface
  SCM_Module *iface = scm_ensure_public_interface(module);
  // Sync this specific symbol into the interface
  SCM *val = module_obarray_lookup(module, sym);
  if (val) {
    scm_c_hash_set_eq(wrap(iface->obarray), wrap(sym), val);
  }
}
```

- [ ] **Step 3: Replace scm_update_module_public_interface stub**

Replace lines 392-397:
```cpp
void scm_update_module_public_interface(SCM_Module *module) {
  // Force rebuild: clear and recreate
  module->public_interface = nullptr;
  scm_ensure_public_interface(module);
}
```

- [ ] **Step 4: Build and verify compilation**

```bash
ninja -C out/build/pscm-cmake
```
Expected: build succeeds

- [ ] **Step 5: Commit**

```bash
git add src/c/module.cc
git commit -m "feat: create public interface module on export"
```

---

### Task 4: Implement binder call in scm_module_variable

**Files:**
- Modify: `src/c/module.cc:155-170`

- [ ] **Step 1: Replace the binder TODO with a real call**

In `scm_module_variable`, replace lines 163-167 (the TODO block) with:

```cpp
  // 2. Call binder (if exists and not a define operation)
  if (!definep && module->binder) {
    // Binder signature: (binder module sym definep) -> variable | #f
    SCM *args = scm_list3(wrap(module), wrap(sym),
                          scm_bool_from_int(definep ? 0 : 1));
    SCM *result = apply_procedure(&g_env, module->binder, cast<SCM_List>(args));
    if (result && !is_falsy(result)) {
      return result;
    }
  }
```

- [ ] **Step 2: Build and verify compilation**

```bash
ninja -C out/build/pscm-cmake
```
Expected: build succeeds

- [ ] **Step 3: Commit**

```bash
git add src/c/module.cc
git commit -m "feat: invoke module binder on variable lookup miss"
```

---

### Task 5: Add set-module-binder! built-in

**Files:**
- Modify: `src/c/module.cc:1017-1025` (init_modules)

- [ ] **Step 1: Write the C++ handler for set-module-binder!**

Insert before `init_modules`:

```cpp
// (set-module-binder! module proc)
SCM *scm_c_set_module_binder(SCM *module, SCM *proc) {
  if (!is_module(module)) {
    eval_error("set-module-binder!: expected module");
  }
  if (!is_proc(proc)) {
    eval_error("set-module-binder!: expected procedure");
  }
  SCM_Module *mod = cast<SCM_Module>(module);
  mod->binder = cast<SCM_Procedure>(proc);
  return scm_none();
}
```

- [ ] **Step 2: Register the built-in**

In `init_modules`, add after line 1024:
```cpp
scm_define_function("set-module-binder!", 2, 0, 0, scm_c_set_module_binder);
```

- [ ] **Step 3: Build and verify**

```bash
ninja -C out/build/pscm-cmake
```
Expected: build succeeds

- [ ] **Step 4: Commit**

```bash
git add src/c/module.cc
git commit -m "feat: add set-module-binder! built-in"
```

---

### Task 6: Implement autoload mechanism and define-module option parsing

**Files:**
- Modify: `src/c/module.cc:155-170` (scm_module_variable, add autoload check)
- Modify: `src/c/module.cc:649-663` (eval_define_module, handle #:autoload)

- [ ] **Step 1: Add autoload check in scm_module_variable**

After the binder check and before returning #f (after the code added in Task 3), add:

```cpp
  // 3. Check autoload specs
  SCM_List *specs = module->autoload_specs;
  while (specs) {
    SCM *spec = specs->data;
    if (is_pair(spec)) {
      SCM_List *spec_pair = cast<SCM_List>(spec);
      // spec is (module-name . sym-list)
      SCM_List *mod_name = cast<SCM_List>(spec_pair->data);
      SCM_List *sym_list = cast<SCM_List>(spec_pair->next->data);

      // Check if the requested symbol is in sym_list
      SCM_List *s = sym_list;
      while (s) {
        if (s->data && is_sym(s->data) &&
            cast<SCM_Symbol>(s->data)->len == sym->len &&
            memcmp(cast<SCM_Symbol>(s->data)->data, sym->data, sym->len) == 0) {
          // Found: load the module and re-search
          SCM *mod_scm = scm_resolve_module(mod_name);
          if (mod_scm && is_module(mod_scm)) {
            // Add loaded module to current module's uses
            SCM_Module *loaded = cast<SCM_Module>(mod_scm);
            SCM_Module *interface = loaded->public_interface ? loaded->public_interface : loaded;
            SCM_List *new_use = make_list(wrap(interface));
            new_use->next = module->uses;
            module->uses = new_use;

            // Now re-search with the newly added use
            SCM *var = module_search_variable(module, sym);
            if (var) return var;
          }
          goto not_found;
        }
        s = s->next;
      }
    }
    specs = specs->next;
  }
not_found:
```

- [ ] **Step 2: Handle #:autoload and other options in eval_define_module**

Replace the TODO at line 653 in `eval_define_module`. Add the option parsing loop after the `scm_c_hash_set_equal` call (after line 657):
```cpp
  // Process options
  SCM_List *opts = options;
  while (opts) {
    SCM *kw = opts->data;
    if (is_sym(kw)) {
      const char *kw_name = cast<SCM_Symbol>(kw)->data;

      if (strcmp(kw_name, "use-module") == 0 && opts->next) {
        // #:use-module (mod-name)
        SCM *mod_name = opts->next->data;
        if (is_pair(mod_name)) {
          SCM *resolved = scm_resolve_module(cast<SCM_List>(mod_name));
          if (resolved && is_module(resolved)) {
            SCM_Module *use_mod = cast<SCM_Module>(resolved);
            SCM_Module *iface = use_mod->public_interface ? use_mod->public_interface : use_mod;
            SCM_List *node = make_list(wrap(iface));
            node->next = cast<SCM_Module>(module_scm)->uses;
            cast<SCM_Module>(module_scm)->uses = node;
          }
        }
        opts = opts->next;
      }
      else if (strcmp(kw_name, "export") == 0) {
        // #:export sym ...
        opts = opts->next;
        while (opts && opts->data && is_sym(opts->data)) {
          scm_module_export(cast<SCM_Module>(module_scm),
                            cast<SCM_Symbol>(opts->data));
          opts = opts->next;
        }
        continue; // already advanced
      }
      else if (strcmp(kw_name, "pure") == 0) {
        cast<SCM_Module>(module_scm)->kind = make_sym("pure");
      }
      else if (strcmp(kw_name, "autoload") == 0 && opts->next) {
        // #:autoload (mod-name sym ...)
        SCM *spec = opts->next->data;
        if (is_pair(spec)) {
          SCM_List *spec_list = cast<SCM_List>(spec);
          if (spec_list->data && is_pair(spec_list->data) && spec_list->next) {
            SCM_List *mod_name = cast<SCM_List>(spec_list->data);
            SCM_List *sym_list = spec_list->next;
            // Build autoload spec: (mod-name . sym-list)
            SCM_List *autoload_spec = make_list(wrap(mod_name));
            autoload_spec->next = make_list(wrap(sym_list));
            autoload_spec->is_dotted = false;
            // Prepend to module's autoload_specs
            SCM_List *node = make_list(wrap(autoload_spec));
            node->next = cast<SCM_Module>(module_scm)->autoload_specs;
            cast<SCM_Module>(module_scm)->autoload_specs = node;
          }
        }
        opts = opts->next;
      }
      else if (strcmp(kw_name, "no-backtrace") == 0) {
        // Flag stored for future use in error.cc
        cast<SCM_Module>(module_scm)->kind = make_sym("no-backtrace");
      }
    }
    opts = opts->next;
  }
```

- [ ] **Step 3: Build and verify**

```bash
ninja -C out/build/pscm-cmake
```
Expected: build succeeds

- [ ] **Step 4: Commit**

```bash
git add src/c/module.cc
git commit -m "feat: implement autoload and define-module option parsing"
```

---

### Task 7: Write public interface isolation test

**Files:**
- Create: `test/module/public_interface_isolation.scm`

- [ ] **Step 1: Write the test**

```scheme
;; Public interface isolation test
;; Verify that use-modules only imports exported symbols

(display "Test 1: define-module with export\n")
(define-module (isolation-mod))
(define pub-x 10)
(define priv-y 20)
(export pub-x)

(display "PASS: define-module and export\n")

(display "\nTest 2: use-modules imports exported symbol\n")
(define-module (isolation-consumer))
(use-modules (isolation-mod))
(if (= pub-x 10)
    (display "PASS: use-modules imports exported symbol\n")
    (begin
      (display "FAIL: use-modules does not import exported symbol\n")
      (exit 1)))

(display "\nTest 3: use-modules does not import non-exported symbol\n")
;; priv-y was not exported, so it should be unbound
(define priv-accessible #t)
(set! priv-accessible
  (catch #t
    (lambda () priv-y #t)
    (lambda (key . args) #f)))
(if (not priv-accessible)
    (display "PASS: non-exported symbol is not accessible\n")
    (begin
      (display "FAIL: non-exported symbol is accessible when it shouldn't be\n")
      (exit 1)))

(display "\nAll public interface isolation tests passed!\n")
```

- [ ] **Step 2: Register test in CMakeLists.txt**

In `test/CMakeLists.txt`, add after the module_use_tests block (after line 124):

```cmake
add_test(
    NAME public_interface_isolation
    COMMAND $<TARGET_FILE:pscm_cc> --test public_interface_isolation.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module
)
```

- [ ] **Step 3: Build and run the test**

```bash
ninja -C out/build/pscm-cmake && ctest --test-dir out/build/pscm-cmake -R public_interface_isolation --output-on-failure
```
Expected: test passes

- [ ] **Step 4: Commit**

```bash
git add test/module/public_interface_isolation.scm test/CMakeLists.txt
git commit -m "test: add public interface isolation test"
```

---

### Task 8: Write binder test

**Files:**
- Create: `test/module/binder_custom.scm`

- [ ] **Step 1: Write the test**

```scheme
;; Custom binder test
;; Verify set-module-binder! and binder invocation

(display "Test 1: create module and set binder\n")
(define-module (binder-test))
(define my-binder
  (lambda (module sym definep)
    (display "binder called for: ") (display sym) (newline)
    (if (eq? sym 'dynamic-val)
        (make-variable 999)
        #f)))
(set-module-binder! (current-module) my-binder)
(display "PASS: set-module-binder!\n")

(display "\nTest 2: binder provides dynamic-val\n")
(if (= dynamic-val 999)
    (display "PASS: binder provides dynamic-val\n")
    (begin
      (display "FAIL: binder did not provide dynamic-val\n")
      (exit 1)))

(display "\nTest 3: binder returns #f for unknown symbol\n")
(define unknown-accessible #t)
(set! unknown-accessible
  (catch #t
    (lambda () undefined-sym #t)
    (lambda (key . args) #f)))
(if (not unknown-accessible)
    (display "PASS: unknown symbol not provided by binder\n")
    (begin
      (display "FAIL: binder returned something for unknown symbol\n")
      (exit 1)))

(display "\nAll binder tests passed!\n")
```

- [ ] **Step 2: Register test in CMakeLists.txt**

Add after the public_interface_isolation block:

```cmake
add_test(
    NAME binder_custom
    COMMAND $<TARGET_FILE:pscm_cc> --test binder_custom.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module
)
```

- [ ] **Step 3: Build and run**

```bash
ninja -C out/build/pscm-cmake && ctest --test-dir out/build/pscm-cmake -R binder_custom --output-on-failure
```
Expected: test passes

- [ ] **Step 4: Commit**

```bash
git add test/module/binder_custom.scm test/CMakeLists.txt
git commit -m "test: add custom binder test"
```

---

### Task 9: Write autoload test

**Files:**
- Create: `test/module/autoload_target.scm` (helper module)
- Create: `test/module/autoload_lazy.scm` (test driver)

- [ ] **Step 1: Write the autoload target module**

`test/module/autoload_target.scm`:
```scheme
(define-module (autoload-target))
(define-public lazy-func
  (lambda (x) (* x 3)))
(display "autoload-target module loaded\n")
```

- [ ] **Step 2: Write the autoload test**

`test/module/autoload_lazy.scm`:
```scheme
;; Autoload lazy loading test
;; Verify that #:autoload only loads module on first access

;; Ensure current directory is in load path so autoload-target.scm is found
(set! %load-path (cons "." %load-path))

(display "Test 1: define module with autoload\n")
(define-module (autoload-consumer)
  #:autoload (autoload-target) lazy-func)

(display "PASS: define-module with #:autoload\n")

(display "\nTest 2: lazy-func should work (triggers autoload)\n")
(if (= (lazy-func 7) 21)
    (display "PASS: autoload lazy-func works\n")
    (begin
      (display "FAIL: autoload lazy-func does not work\n")
      (exit 1)))

(display "\nTest 3: lazy-func available after autoload\n")
(if (= (lazy-func 2) 6)
    (display "PASS: autoload symbol persists\n")
    (begin
      (display "FAIL: autoload symbol did not persist\n")
      (exit 1)))

(display "\nAll autoload tests passed!\n")
```

- [ ] **Step 3: Register test in CMakeLists.txt**

```cmake
add_test(
    NAME autoload_lazy
    COMMAND $<TARGET_FILE:pscm_cc> --test autoload_lazy.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module
)
```

- [ ] **Step 4: Build and run**

```bash
ninja -C out/build/pscm-cmake && ctest --test-dir out/build/pscm-cmake -R autoload_lazy --output-on-failure
```
Expected: test passes

- [ ] **Step 5: Commit**

```bash
git add test/module/autoload_target.scm test/module/autoload_lazy.scm test/CMakeLists.txt
git commit -m "test: add autoload lazy loading test"
```

---

### Task 10: Write define-module options test

**Files:**
- Create: `test/module/define_module_options.scm`

- [ ] **Step 1: Write the test**

```scheme
;; define-module option parsing test
;; Verify #:use-module and #:export inside define-module body

(display "Test 1: define-module with #:use-module\n")
(define-module (options-source))
(define-public src-val 42)
(define-public (src-func x) (+ x 1))

(define-module (options-consumer)
  #:use-module (options-source)
  #:export consumer-val)

(display "PASS: define-module with options\n")

(display "\nTest 2: #:use-module imports bindings\n")
(if (= src-val 42)
    (display "PASS: #:use-module imports variable\n")
    (begin
      (display "FAIL: #:use-module did not import variable\n")
      (exit 1)))

(if (= (src-func 5) 6)
    (display "PASS: #:use-module imports function\n")
    (begin
      (display "FAIL: #:use-module did not import function\n")
      (exit 1)))

(display "\nTest 3: #:pure flag\n")
(define-module (pure-test) #:pure)
(display "PASS: define-module with #:pure\n")

(display "\nAll define-module options tests passed!\n")
```

- [ ] **Step 2: Register test in CMakeLists.txt**

```cmake
add_test(
    NAME define_module_options
    COMMAND $<TARGET_FILE:pscm_cc> --test define_module_options.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module
)
```

- [ ] **Step 3: Build and run**

```bash
ninja -C out/build/pscm-cmake && ctest --test-dir out/build/pscm-cmake -R define_module_options --output-on-failure
```
Expected: test passes

- [ ] **Step 4: Commit**

```bash
git add test/module/define_module_options.scm test/CMakeLists.txt
git commit -m "test: add define-module option parsing test"
```

---

### Task 11: Full regression verification

- [ ] **Step 1: Run all lit test suites**

```bash
ninja -C out/build/pscm-cmake check-base check-cont check-core check-sicp check-macro check-gc
```
Expected: all pass (100%)

- [ ] **Step 2: Run all CTest module tests**

```bash
ctest --test-dir out/build/pscm-cmake -R "module|autoload|binder|public_interface" --output-on-failure
```
Expected: all pass

- [ ] **Step 3: Run existing module tests to ensure no regressions**

```bash
ctest --test-dir out/build/pscm-cmake -R "module" --output-on-failure
```
Expected: all existing module tests still pass

- [ ] **Step 4: Commit if any fixes were needed, otherwise done**
