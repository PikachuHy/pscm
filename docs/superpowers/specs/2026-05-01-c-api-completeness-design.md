# C API Completeness Design

**Date**: 2026-05-01
**Goal**: Bring pscm's C API to parity with Guile 1.8's core embedding API, in three phases: A (call/define), B (module), C (port control).

## Architecture

All new functions live in existing files — no new translation units:

- `pscm_api.cc` / `pscm_api.h`: public C API (`pscm_*` prefix)
- `pscm.h` / corresponding `.cc`: internal/scm-level API (`scm_*` prefix)

The `pscm_*` wrappers in `pscm_api.cc` delegate to `scm_*` functions.

## Phase A: Core Embedding API

### `scm_c_define`

```c
SCM *scm_c_define(const char *name, SCM *val);
```

- Look up `name` in current module via `scm_c_lookup`
- If variable exists: `scm_variable_set_x(var, val)`
- If not: `scm_make_variable(val)`, insert into current module's obarray
- Returns the variable object

### `scm_c_define_gsubr`

```c
SCM *scm_c_define_gsubr(const char *name, int req, int opt, int rst,
                        SCM *(*fcn)(void));
```

- Creates an `SCM_Function` with `n_args` derived from `(req, opt, rst)`
- Calls `scm_c_define(name, proc)` to register in current module
- This is a C-linkage wrapper around the existing template `scm_define_function`

### `pscm_c_define` / `pscm_c_define_gsubr`

Thin `extern "C"` wrappers in `pscm_api.cc` that delegate to the `scm_*` versions.

### Already exists (no work needed)

- `scm_call_0/1/2/3` — call Scheme procedure from C
- `scm_apply_0` — apply with arg list
- `pscm_eval_string` — eval C string

## Phase B: Module System C API

### New functions

```c
// Resolve module by string name "(foo bar)" (overload of existing SCM* version)
SCM *scm_c_resolve_module(const char *name);

// Lookup/define in a specific module
SCM *scm_c_module_lookup(SCM *module, const char *name);
SCM *scm_c_module_define(SCM *module, const char *name, SCM *val);

// Scheme-level module operations (expose existing internals as public)
SCM *scm_module_lookup(SCM *module, SCM *sym);
SCM *scm_module_define(SCM *module, SCM *sym, SCM *val);

// Create module with optional init callback
SCM *scm_c_define_module(const char *name,
                         void (*init)(void *), void *data);

// Temporarily switch current module, run func, restore on exit
SCM *scm_c_call_with_current_module(SCM *module,
                                    SCM *(*func)(void *), void *data);

// Use (import) a module from C
void scm_c_use_module(const char *name);
```

### Implementation notes

- `scm_c_resolve_module(const char*)`: parse string → symbol list → call existing `scm_c_resolve_module(SCM*)`
- `scm_c_module_lookup`: `module_search_variable` + `scm_variable_ref`
- `scm_c_module_define`: insert/update binding in module obarray
- `scm_c_define_module`: resolve (or create) module → `scm_c_call_with_current_module` + init
- `scm_c_call_with_current_module`: save current → `scm_set_current_module(new)` → run func → restore (use catch for safety)

## Phase C: Port Control API

### New functions

```c
SCM *scm_current_input_port(void);
SCM *scm_set_current_input_port(SCM *port);
SCM *scm_current_output_port(void);
SCM *scm_set_current_output_port(SCM *port);
```

Plus corresponding `pscm_*` wrappers in `pscm_api.cc`.

### Implementation

Follow the existing `g_current_error_port` pattern:
- Two `static SCM*` globals in `port.cc`: `g_current_input_port`, `g_current_output_port`
- Default to stdin/stdout file ports, initialized in `init_port()`
- Register as Scheme builtins (`current-input-port`, etc.)

## Testing

- New C++ unit tests in `test/unit/` or expand `test/module/` for module C API
- New Scheme lit tests for each new API function
- Smoke test: C host calls `pscm_init()` → `pscm_c_define("x", int)` → `pscm_eval_string("x")` → verify round-trip

## Scope notes

- Not implementing: `scm_c_make_gsubr` (complex subr type system), `scm_dynwind_*_port` (requires fluids/dynamic state)
- The `pscm_*` wrappers provide `extern "C"` linkage suitable for embedding in C programs
- No new data structures — all additions are API wrappers over existing internals
