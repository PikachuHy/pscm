# Module Environment Chain — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bake the defining module into SCM_Environment so `make_proc` naturally captures module-private bindings through `proc->env`, matching Guile 1.8's eval closure model.

**Architecture:** Add `SCM_Environment::module` pointer. When loading a module file, create a module-aware environment whose parent chain includes the module's obarray. Procedures defined inside the module capture this environment. During macro expansion, the transformer's proc->env already chains through the defining module — no special-case code needed.

**Tech Stack:** C++20, pscm codebase

---

### Task 1: Add module field to SCM_Environment and update make_env

**Files:**
- Modify: `src/c/pscm_types.h:179-181`
- Modify: `src/c/pscm.h:194-200`

- [ ] **Step 1: Add `module` field to SCM_Environment**

In `/Users/pikachu/pr/pscm/src/c/pscm_types.h`, add the module pointer after the `parent` field:

```cpp
// Current (lines 179-181):
  List dummy;
  SCM_Environment *parent;
};

// Replace with:
  List dummy;
  SCM_Environment *parent;
  SCM_Module *module;  // if non-null, search this module's obarray+uses during lookup
};
```

- [ ] **Step 2: Update make_env to propagate module**

In `/Users/pikachu/pr/pscm/src/c/pscm.h`, update the `make_env` inline function at line 194:

```cpp
// Current (lines 194-200):
inline SCM_Environment *make_env(SCM_Environment *parent) {
  auto env = (SCM_Environment *)gc_alloc(GC_ENV, sizeof(SCM_Environment));
  env->parent = parent;
  env->dummy.data = nullptr;
  env->dummy.next = nullptr;
  return env;
}

// Replace with:
inline SCM_Environment *make_env(SCM_Environment *parent) {
  auto env = (SCM_Environment *)gc_alloc(GC_ENV, sizeof(SCM_Environment));
  env->parent = parent;
  env->module = parent ? parent->module : nullptr;
  env->dummy.data = nullptr;
  env->dummy.next = nullptr;
  return env;
}
```

- [ ] **Step 3: Add make_module_environment**

After `make_env` in pscm.h, add the new function:

```cpp
// Create a module-aware top-level environment.  When expressions are
// evaluated in this environment, lookups traverse the module's obarray
// and use chain before falling through to the global environment.
inline SCM_Environment *make_module_environment(SCM_Module *mod, SCM_Environment *parent) {
  auto env = make_env(parent);
  env->module = mod;
  return env;
}
```

- [ ] **Step 4: Build and verify**

```bash
ninja -C /Users/pikachu/pr/pscm/out/build/pscm-cmake
```

Expected: builds successfully (new field is unused but initialized to nullptr by existing `make_env(nullptr)` calls).

- [ ] **Step 5: Commit**

```bash
cd /Users/pikachu/pr/pscm && git add src/c/pscm_types.h src/c/pscm.h && git commit -m "feat: add module pointer to SCM_Environment with make_module_environment"
```

---

### Task 2: Add module search to scm_env_search and revert step-5 fallback

**Files:**
- Modify: `src/c/environment.cc:57-122`

- [ ] **Step 1: Add module search between step 2 and step 3**

In `/Users/pikachu/pr/pscm/src/c/environment.cc`, add a new step 2.5 after the parent environment search (after line 71, before the "3. If current module exists" comment at line 73):

```cpp
  // 2.5. Search the module baked into the environment chain (if any).
  // This finds module-private variables (e.g. cur-props in tm-define.scm)
  // during evaluation of code loaded inside that module.  The module is
  // set by make_module_environment when the module's file is loaded.
  // We walk the parent chain to find the first non-null module pointer;
  // child environments inherit it from make_module_environment via make_env.
  {
    SCM_Environment *e = env;
    while (e) {
      if (e->module) {
        SCM *var = module_search_variable(e->module, sym);
        if (var) {
          return var;
        }
        break;  // found a module but symbol wasn't in it; don't check again
      }
      e = e->parent;
    }
  }
```

Place this before the "3. If current module exists" comment block (current line 73).

- [ ] **Step 2: Revert step-5 g_macro_defining_module fallback**

Remove lines 97-108 (the `extern SCM *g_macro_defining_module` block and associated comment). This is no longer needed — module-private variables are now found through the environment chain (step 2.5 above).

- [ ] **Step 3: Build and run tests**

```bash
ninja -C /Users/pikachu/pr/pscm/out/build/pscm-cmake
ctest --test-dir /Users/pikachu/pr/pscm/out/build/pscm-cmake --output-on-failure
```

Expected: 38/38 tests pass (100%). The new module search only fires when `env->module` is non-null, which only happens for code loaded via `scm_resolve_module` — no existing test uses that path yet.

- [ ] **Step 4: Commit**

```bash
cd /Users/pikachu/pr/pscm && git add src/c/environment.cc && git commit -m "feat: search environment chain module in scm_env_search; revert g_macro_defining_module fallback"
```

---

### Task 3: Add scm_c_primitive_load_with_env

**Files:**
- Modify: `src/c/load.cc:23-35`

- [ ] **Step 1: Add the new function**

In `/Users/pikachu/pr/pscm/src/c/load.cc`, add after the existing `scm_c_primitive_load` function (after line 35):

```cpp
// Load a file evaluating each expression in the provided environment.
// Unlike scm_c_primitive_load (which always uses &g_env), this allows
// loading module files in a module-aware environment so define forms
// capture the module's obarray in their procedure environments.
SCM *scm_c_primitive_load_with_env(const char *filename, SCM_Environment *env) {
  fprintf(stderr, "[load] Loading file: %s\n", filename);

  SCM_List *expr_list = parse_file(filename);
  if (!expr_list) {
    eval_error("primitive-load: failed to load file: %s", filename);
    return nullptr;
  }

  SCM *result = scm_none();
  SCM_List *it = expr_list;
  while (it) {
    result = eval_with_env(env, it->data);
    it = it->next;
  }

  return result;
}
```

- [ ] **Step 2: Declare the function**

In `/Users/pikachu/pr/pscm/src/c/pscm.h`, add the declaration near the other load declarations (search for `scm_c_primitive_load`). Add:

```cpp
SCM *scm_c_primitive_load_with_env(const char *filename, SCM_Environment *env);
```

- [ ] **Step 3: Build**

```bash
ninja -C /Users/pikachu/pr/pscm/out/build/pscm-cmake
```

Expected: builds successfully.

- [ ] **Step 4: Commit**

```bash
cd /Users/pikachu/pr/pscm && git add src/c/load.cc src/c/pscm.h && git commit -m "feat: add scm_c_primitive_load_with_env for module-aware file loading"
```

---

### Task 4: Use module environment in scm_resolve_module

**Files:**
- Modify: `src/c/module.cc:398-414`

- [ ] **Step 1: Switch to module-aware loading**

In `/Users/pikachu/pr/pscm/src/c/module.cc`, replace the file loading block (lines 402-413):

```cpp
// Current (lines 402-413):
  if (found_path && !module_exists) {
    // Save current module
    SCM *old_module = scm_current_module();
    
    // Set current module to the target module
    scm_set_current_module(module);
    
    // Load and evaluate file (reuse scm_c_primitive_load)
    scm_c_primitive_load(found_path);
    
    // Restore current module
    scm_set_current_module(old_module);
    
    if (full_path) {
      free(full_path);
    }
  }

// Replace with:
  if (found_path && !module_exists) {
    // Save current module
    SCM *old_module = scm_current_module();
    
    // Create a module-aware environment so procedures defined in this
    // file capture the module's obarray in their environment chain.
    SCM_Module *mod = cast<SCM_Module>(module);
    SCM_Environment *module_env = make_module_environment(mod, &g_env);
    
    // Set current module for define forms (they check scm_current_module)
    scm_set_current_module(module);
    
    // Load and evaluate file in module-aware environment
    scm_c_primitive_load_with_env(found_path, module_env);
    
    // Restore current module
    scm_set_current_module(old_module);
    
    if (full_path) {
      free(full_path);
    }
  }
```

- [ ] **Step 2: Build and run tests**

```bash
ninja -C /Users/pikachu/pr/pscm/out/build/pscm-cmake
ctest --test-dir /Users/pikachu/pr/pscm/out/build/pscm-cmake --output-on-failure
```

Expected: 38/38 tests pass. Module tests (`nested_module`, `autoload_lazy`, `module_texmacs_module`, etc.) should still pass — the module env only adds lookup capability, it doesn't change existing behavior.

- [ ] **Step 3: Commit**

```bash
cd /Users/pikachu/pr/pscm && git add src/c/module.cc && git commit -m "feat: use module-aware environment when loading module files"
```

---

### Task 5: Add GC tracing for env->module

**Files:**
- Modify: `src/c/gc.cc` — `trace_env` function

- [ ] **Step 1: Add trace_ptr for module**

In `/Users/pikachu/pr/pscm/src/c/gc.cc`, find `trace_env` (search for `static void trace_env`). After the line `trace_ptr(env->parent, stack);`, add:

```cpp
  trace_ptr(env->module, stack);
```

The updated function:

```cpp
static void trace_env(GCBlock *block, MarkStack *stack) {
  SCM_Environment *env = (SCM_Environment *)block_to_obj(block);

  SCM_Environment::List *list = env->dummy.next;
  while (list) {
    if (list->data) {
      trace_ptr(list->data->value, stack);
    }
    list = list->next;
  }

  trace_ptr(env->parent, stack);
  trace_ptr(env->module, stack);
}
```

- [ ] **Step 2: Build and run GC tests**

```bash
ninja -C /Users/pikachu/pr/pscm/out/build/pscm-cmake
ctest --test-dir /Users/pikachu/pr/pscm/out/build/pscm-cmake -R gc --output-on-failure
```

Expected: GC tests pass. The new trace prevents modules from being collected while environments reference them.

- [ ] **Step 3: Commit**

```bash
cd /Users/pikachu/pr/pscm && git add src/c/gc.cc && git commit -m "fix: trace env->module in GC to prevent module collection"
```

---

### Task 6: Revert macro.cc and pscm_types.h workaround changes

**Files:**
- Modify: `src/c/macro.cc:1-7` (remove g_macro_defining_module)
- Modify: `src/c/macro.cc:126-137` (revert expand_macro_call module save/restore)
- Modify: `src/c/pscm_types.h:101-106` (remove defining_module from SCM_Macro)
- Modify: `src/c/gc.cc:634-640` (revert trace_macro addition)

- [ ] **Step 1: Revert g_macro_defining_module global**

In `/Users/pikachu/pr/pscm/src/c/macro.cc`, remove lines 4-7:

```cpp
// REMOVE these lines:
// During macro expansion, set to the module where the macro was defined.
// scm_env_search checks this as a fallback after the global environment so
// module-private variables are visible but don't shadow global bindings.
SCM *g_macro_defining_module = nullptr;
```

- [ ] **Step 2: Revert expand_macro_call module save/restore**

In `/Users/pikachu/pr/pscm/src/c/macro.cc`, replace lines 126-137:

```cpp
// Current (lines 126-137):
  // Set the defining module as a fallback lookup target.  Module-private
  // variables (e.g. property-rewrite in tm-define.scm) need to be visible
  // during macro expansion, but should NOT shadow global stubs or the
  // expansion-site module.  The defining module is searched last.
  SCM *saved_defining_module = g_macro_defining_module;
  g_macro_defining_module = macro->defining_module;

  // Call the macro transformer (evaluate in macro environment)
  SCM *expanded = eval_with_list(macro_env, macro->transformer->body);

  // Restore
  g_macro_defining_module = saved_defining_module;

// Replace with:
  // Call the macro transformer (evaluate in macro environment).
  // The transformer's proc->env was set to a module-aware environment
  // by make_proc, so module-private variables are found through the
  // normal environment parent chain during eval_with_list.
  SCM *expanded = eval_with_list(macro_env, macro->transformer->body);
```

- [ ] **Step 3: Remove defining_module from SCM_Macro**

In `/Users/pikachu/pr/pscm/src/c/pscm_types.h`, remove the `defining_module` field:

```cpp
// Current (lines 101-106):
struct SCM_Macro {
  SCM_Symbol *name;
  SCM_Procedure *transformer; // Macro transformer procedure
  SCM_Environment *env;       // Environment where macro was defined
  SCM *defining_module;       // Module where macro was defined (for module-scoped lookup)
};

// Replace with:
struct SCM_Macro {
  SCM_Symbol *name;
  SCM_Procedure *transformer; // Macro transformer procedure
  SCM_Environment *env;       // Environment where macro was defined
};
```

- [ ] **Step 4: Revert trace_macro addition**

In `/Users/pikachu/pr/pscm/src/c/gc.cc`, find `trace_macro` (line 635):

```cpp
// Current:
static void trace_macro(GCBlock *block, MarkStack *stack) {
  SCM_Macro *macro = (SCM_Macro *)block_to_obj(block);
  trace_ptr(macro->name,             stack); // SCM_Symbol*
  trace_ptr(macro->transformer,      stack); // SCM_Procedure*
  trace_ptr(macro->env,              stack); // SCM_Environment*
  trace_ptr(macro->defining_module,  stack); // SCM_Module*
}

// Replace with:
static void trace_macro(GCBlock *block, MarkStack *stack) {
  SCM_Macro *macro = (SCM_Macro *)block_to_obj(block);
  trace_ptr(macro->name,        stack); // SCM_Symbol*
  trace_ptr(macro->transformer, stack); // SCM_Procedure*
  trace_ptr(macro->env,         stack); // SCM_Environment*
}
```

- [ ] **Step 5: Build and run full test suite**

```bash
ninja -C /Users/pikachu/pr/pscm/out/build/pscm-cmake
ctest --test-dir /Users/pikachu/pr/pscm/out/build/pscm-cmake --output-on-failure
```

Expected: 38/38 tests pass (100%).

- [ ] **Step 6: Commit**

```bash
cd /Users/pikachu/pr/pscm && git add src/c/macro.cc src/c/pscm_types.h src/c/gc.cc && git commit -m "revert: remove g_macro_defining_module workaround, superseded by environment chain module"
```

---

### Task 7: Verify TeXmacs loading regressions and write module-scope test

**Files:**
- Create: `test/module/module_private_macro.scm` — test that module-private variables are visible during macro expansion
- Create: `test/module/Output/module_private_macro.script` — test expectations
- Modify: `test/module/texmacs/load_texmacs_init.scm` — remove global stubs that should no longer be needed

- [ ] **Step 1: Write the module-private macro test**

Create `/Users/pikachu/pr/pscm/test/module/module_private_macro.scm`:

```scheme
;; Test: module-private variables are visible during macro expansion.
;; This verifies the environment chain module fix — macro transformers
;; defined inside a module can reference that module's private (define) bindings.

(define-module (test macro-private) #:export (use-private))

;; Module-private variable — NOT exported
(define secret-value "module-secret")

;; Module-private function — NOT exported
(define (make-message x)
  (string-append "MSG:" x))

;; Macro whose transformer references module-private bindings
(define-macro (use-private)
  `(string-append ,make-message ,secret-value))

;; Now use the macro (within the same module)
(display (use-private)) (newline)
(display "PASS") (newline)
```

Create `/Users/pikachu/pr/pscm/test/module/Output/module_private_macro.script`:

```
RUN: %pscm_cc --test %S/../module_private_macro.scm 2>&1 | %filter
CHECK: MSG:module-secret
CHECK: PASS
```

Register in `/Users/pikachu/pr/pscm/test/CMakeLists.txt` (follow existing pattern around line 154):

```cmake
add_test(
    NAME module_private_macro
    COMMAND $<TARGET_FILE:pscm_cc> --test module_private_macro.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module
)
```

- [ ] **Step 2: Build and run the new test**

```bash
ninja -C /Users/pikachu/pr/pscm/out/build/pscm-cmake
ctest --test-dir /Users/pikachu/pr/pscm/out/build/pscm-cmake -R module_private_macro --output-on-failure
```

Expected: PASS. The macro transformer references `make-message` and `secret-value` — both module-private — and they're found through the environment chain.

- [ ] **Step 3: Run TeXmacs loading test**

```bash
/Users/pikachu/pr/pscm/out/build/pscm-cmake/pscm_cc --test /Users/pikachu/pr/pscm/test/module/texmacs/load_texmacs_init.scm 2>&1 | tail -5
```

Expected: Should progress further than before. Module-private variables in tm-define.scm are now visible during macro expansion without needing global stubs.

- [ ] **Step 4: Run full test suite**

```bash
ctest --test-dir /Users/pikachu/pr/pscm/out/build/pscm-cmake --output-on-failure
```

Expected: 39/39 tests pass (100%).

- [ ] **Step 5: Commit**

```bash
cd /Users/pikachu/pr/pscm && git add test/module/module_private_macro.scm test/module/Output/module_private_macro.script test/CMakeLists.txt && git commit -m "test: add module-private macro expansion test"
```

---

### Summary

| Task | Files | Purpose |
|------|-------|---------|
| 1 | `pscm_types.h`, `pscm.h` | Add `module` field + `make_module_environment` |
| 2 | `environment.cc` | Search env chain module; revert step-5 |
| 3 | `load.cc`, `pscm.h` | `scm_c_primitive_load_with_env` |
| 4 | `module.cc` | Use module env in `scm_resolve_module` |
| 5 | `gc.cc` | Trace `env->module` |
| 6 | `macro.cc`, `pscm_types.h`, `gc.cc` | Revert g_macro_defining_module workaround |
| 7 | `test/module/` | Integration test + verify TeXmacs loading |
