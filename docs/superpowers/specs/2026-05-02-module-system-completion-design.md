# Module System Completion Design

## Context

pscm's module system has the foundational data model in place (`SCM_Module`
already carries `binder`, `public_interface`, `exports`) but three logical gaps
block Guile 1.8 compatibility:

1. **Public interface module is never created** — exports are tracked in a
   list, but `use-modules` / `module-use!` fall back to the raw module because
   `public_interface` is always null.
2. **Binder is never called** — the field exists but `scm_module_variable`
   returns `#f` without invoking the binder. Autoload depends on this.
3. **define-module option parsing is a stub** — `#:use-module`, `#:export`,
   `#:autoload`, `#:pure` are ignored.

## Design

### 1. Public Interface Module

Add one new internal function and modify one existing function in `module.cc`.

**`scm_ensure_public_interface(SCM_Module *module) → SCM_Module*`**

```
if module->public_interface already exists:
  return module->public_interface

1. Create a new SCM_Module via scm_make_module(name, 31)
2. Set its kind to 'interface
3. Iterate module->exports:
   for each symbol, copy the binding from module->obarray
   into interface->obarray
4. module->public_interface = interface
5. interface->public_interface = module   (bidirectional, Guile convention)
return interface
```

**Modify `scm_module_export`:**

After adding the symbol to `module->exports` (existing behaviour), also call
`scm_ensure_public_interface` to sync the new export into the interface module's
obarray.

**Downstream effect:** The existing fallback code in `eval_use_modules`
(line 682), `scm_c_module_use` (line 905), and `scm_c_use_module` (line 980)
already checks `module->public_interface` and prefers it over the raw module.
Once the interface module actually exists, export isolation takes effect with no
further changes to those call sites.

### 2. Binder Call + Autoload

**Binder call in `scm_module_variable`:**

Replace the TODO comment at line 165 with:

```
if (!definep && module->binder):
  SCM *args = scm_list3(wrap(module), wrap(sym),
                        scm_bool_from_int(!definep))
  SCM *result = apply_procedure(module->binder, args)
  if result is not #f:
    return result
```

The binder receives `(module, symbol, definep)` and returns a variable or `#f`.

**`make_autoload_binder(module_name_list, symbol_list) → SCM_Procedure*`**

Creates a C++ lambda wrapped as `SCM_Procedure` that:

1. Checks whether the requested symbol is in `symbol_list`.
2. If yes: calls `scm_resolve_module(module_name_list)` to load the target
   module, then searches it for the symbol and returns the variable.
3. If no: returns `#f`.

The binder's module argument is ignored (the autoload target is baked in at
creation time).

### 3. define-module Option Parsing

Rewrite the option-processing block in `eval_define_module` (line 652) to
iterate `options` and dispatch on keyword:

| Keyword | Action |
|---|---|
| `#:use-module mod-name` | `scm_resolve_module` → prepend to module's `uses` |
| `#:export sym ...` | Call `scm_module_export` for each symbol |
| `#:pure` | Set `module->kind = make_sym("pure")` |
| `#:autoload (mod-name sym ...)` | Call `make_autoload_binder`, set as `module->binder` |
| `#:no-backtrace` | Set a flag on the module (future: honour in error.cc) |
| Unknown keyword | Warn to stderr, skip (forward-compatible) |

Order: process `#:use-module` first so exported symbols from used modules are
visible, then `#:export`, then `#:autoload` and other options.

## Tests

Add to `test/module/`:

| File | What it verifies |
|---|---|
| `public_interface_isolation.scm` | Export `foo` but not `bar`; use-modules sees `foo`, `bar` is unbound |
| `autoload_lazy.scm` | `#:autoload` symbols trigger file load only on first access |
| `binder_custom.scm` | `set-module-binder!` with a custom binder procedure |
| `define_module_options.scm` | `#:use-module`, `#:export`, `#:pure` inside `define-module` body |

## Non-goals

- `#:renamer` and `#:duplicates` — rarely used in practice, deferred
- `#:use-syntax` / `#:export-syntax` — requires syntax-object support, deferred
- Thread safety — the entire pscm runtime is single-threaded; this doesn't change that
