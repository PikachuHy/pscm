# Module Environment Chain — Guile 1.8 Eval Closure Design

**Date**: 2026-05-05
**Goal**: Bake the defining module's obarray into the procedure environment chain so macro transformers naturally capture module-private bindings, matching Guile 1.8's eval closure model.

## Problem

When a macro is defined inside a module, its transformer procedure captures `proc->env = &g_env` (the global environment). Module-private variables like `cur-props` live in the module's `obarray` hash table, which is not part of the environment chain. They're found only through `scm_env_search` step 3 — which checks the **current** (expansion-site) module, not the **defining** module.

This forces global stubs for `cur-props`, `property-rewrite`, `filter-conds`, and every other tm-define internal that `define-macro` transformers reference.

## Design

Add a `module` pointer to `SCM_Environment`. When creating procedures inside a module, pass a module-aware environment so `proc->env` chains through the module's obarray. Module-private lookups then work naturally through normal environment parent traversal — no special-case code in `scm_env_search`.

### Before

```
make_proc(env=&g_env):
  proc->env = &g_env           ← only globals

apply_procedure(proc):
  proc_env → &g_env            ← no module bindings in chain

scm_env_search:
  step 1: proc_env (params)
  step 2: parent chain (= &g_env)
  step 3: scm_current_module() ← WRONG: expansion-site module
```

### After

```
make_proc(env=module_env):
  proc->env = module_env       ← module + globals

apply_procedure(proc):
  proc_env → module_env → &g_env

scm_env_search_entry traversal:
  proc_env (params)
  module_env → module->obarray → module->uses  ← CORRECT: defining module
  &g_env (globals)
```

### Data structure change

`pscm_types.h` — add one field to `SCM_Environment`:

```cpp
struct SCM_Environment {
  struct Entry { char *key; SCM *value; };
  struct List { Entry *data; List *next; };
  List dummy;
  SCM_Environment *parent;
  SCM_Module *module;  // NEW: if non-null, search obarray+uses during parent traversal
};
```

### Key functions

**`make_env`** — propagate `module` to child environments:

```cpp
inline SCM_Environment *make_env(SCM_Environment *parent) {
  auto env = (SCM_Environment *)gc_alloc(GC_ENV, sizeof(SCM_Environment));
  env->parent = parent;
  env->module = parent ? parent->module : nullptr;
  env->dummy.data = nullptr;
  env->dummy.next = nullptr;
  return env;
}
```

**`make_module_environment`** — create a top-level environment for a module:

```cpp
inline SCM_Environment *make_module_environment(SCM_Module *mod, SCM_Environment *parent) {
  auto env = make_env(parent);
  env->module = mod;
  return env;
}
```

**`scm_env_search_entry`** — add module search during parent traversal. The module lookup returns `SCM*` (not `Entry*`), so the search is added in `scm_env_search` (which wraps `scm_env_search_entry`) as a new step 2.5 between parent-env search (step 2) and current-module search (step 3). The logic: after exhausting the lexical parent chain, if any environment in the chain had `module` set, search that module's obarray + uses via `module_search_variable`. If found, return the value.

**`scm_env_search` revised order:**
1. Local frame (`scm_env_exist`)
2. Parent chain (`scm_env_search_entry` with `search_parent=true`)
3. **NEW: if any env in parent chain has `module`, search `module->obarray + uses`**
4. Current module (`scm_current_module()`)
5. Global env (`&g_env` if `env != &g_env`)

**`scm_c_primitive_load_with_env`** — new function (load.cc). Same as `scm_c_primitive_load` but evaluates expressions in a caller-provided environment instead of always `&g_env`.

**`scm_resolve_module`** — use module environment for file loading (module.cc:400-411):

```cpp
if (found_path && !module_exists) {
    SCM_Environment *module_env = make_module_environment(
        cast<SCM_Module>(module), &g_env);
    scm_set_current_module(module);
    scm_c_primitive_load_with_env(found_path, module_env);
    scm_set_current_module(old_module);
}
```

### GC tracing

`gc.cc` — `trace_env` must trace the new `module` field so the GC doesn't collect a module referenced by an environment:

```cpp
static void trace_env(GCBlock *block, MarkStack *stack) {
  SCM_Environment *env = (SCM_Environment *)block_to_obj(block);
  // ... existing entry tracing ...
  trace_ptr(env->parent, stack);
  trace_ptr(env->module, stack);  // NEW
}
```

### Revert prior workarounds

- Remove `g_macro_defining_module` global and its usage in `expand_macro_call` (macro.cc)
- Remove `SCM_Macro::defining_module` field (pscm_types.h)
- Remove step-5 fallback from `scm_env_search` (environment.cc)
- Revert `trace_macro` addition for `defining_module` (gc.cc)

### Files changed

| File | Change |
|------|--------|
| `pscm_types.h` | `+module` in `SCM_Environment`; `-defining_module` in `SCM_Macro` |
| `pscm.h` | Update `make_env` to propagate `module`; add `make_module_environment` |
| `gc.cc` | Add `trace_ptr(env->module)` in `trace_env`; revert `trace_macro` addition |
| `environment.cc` | Add module search in `scm_env_search_entry`; revert step-5 fallback |
| `load.cc` | Add `scm_c_primitive_load_with_env` |
| `module.cc` | Use `make_module_environment` + `scm_c_primitive_load_with_env` in `scm_resolve_module` |
| `macro.cc` | Revert `g_macro_defining_module` and `expand_macro_call` changes |

### Non-goals

- `module->transformer` (syntax transformation) — not needed for macro expansion
- `module->eval_closure` as a Scheme-visible procedure — the environment chain replaces it internally; the field stays as vestigial for API compatibility
- Changing how `eval_with_env` dispatches special forms — macro lookup in eval.cc (lines 300-340) is unchanged

### Risk

- `module_search_variable` calls the binder and searches the root module — could change behavior for code that relies on the current `scm_env_search` step 3 ordering. Mitigation: `module_search_variable` already exists and is used in `scm_env_search` step 3; this just wires it into the parent chain instead.
- Environments created outside module context (non-module `--test` scripts) continue to use `&g_env` with `module = nullptr`, so they're unaffected.
