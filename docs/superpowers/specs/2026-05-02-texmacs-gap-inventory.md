# TeXmacs Compatibility Gap Inventory

**Date**: 2026-05-02
**Based on**: Incremental loading of TeXmacs init-texmacs.scm through pscm (Phase 3)

## Loading Status

- [x] Loads init-texmacs.scm (main entry)
- [x] Loads kernel/boot/boot.scm
- [x] Loads kernel/boot/compat.scm
- [x] Loads kernel/boot/abbrevs.scm
- [x] Loads kernel/boot/debug.scm
- [x] Loads kernel/boot/srfi.scm
- [x] Loads kernel/library/* (base, list, tree, content, patch, iterator)
- [x] Loads kernel/regexp/* (match, select, test)
- [x] Loads kernel/logic/* (rules, bind, unify, query, data, test)
- [x] Loads kernel/texmacs/* (tm-define, tm-preferences, tm-modes, tm-plugins, tm-language, tm-convert, tm-dialogue, tm-file-system, tm-secure, tm-states)
- [x] Loads kernel/old-gui/* (factory, widget, form, test)
- [x] Loads kernel/gui/* (menu-convert, menu-widget, menu-define, menu-test, gui-markup, kbd-define, kbd-handlers, speech-define)
- [x] Loads kernel/boot/ahash-table.scm, kernel/boot/prologue.scm
- [x] All 45 kernel/ files load without uncaught errors (KERNEL-LOADED milestone: 2026-05-05)
- [x] All 46 utils/ files attempted — 20 pass, 26 caught errors (load-order: macros from not-yet-loaded modules)

## Blocking Issues Found (All Resolved as of 2026-05-05)

| # | Issue | Location | Category | Status |
|---|-------|----------|----------|--------|
| 1 | Special forms not aliaseable as values | boot.scm:100 | Module system | Resolved via scheme-dialect workaround |
| 2 | Module-scoped privates invisible to macro expansion | tm-define.scm:127-168 | Macro/module | Resolved by module environment chain (741919b) |
| 3 | `with` macro stub didn't handle destructuring | kbd-handlers.scm via tm-define | Macro system | Resolved (41f2ef3) |
| 4 | `assert(is_sym(...))` in procedure/macro param binding | procedure.cc:201,167 macro.cc:100 | Error handling | Resolved — replaced with eval_error (41f2ef3) |

## Missing C++ Functions (Stubbed)

These TeXmacs C++ functions were encountered and stubbed. A full TeXmacs integration would need real implementations.

| Function | Stub Behavior | Needed For |
|----------|--------------|------------|
| `url-concretize` | Returns hardcoded boot.scm path | Path resolution for `$TEXMACS_PATH` |
| `gui-version` | Returns `"qt5"` | Platform detection |
| `os-win32?` | Returns `#f` | Platform branching |
| `os-linux?` | Returns `#f` | Platform branching |
| `os-gnu?` | Returns `#f` | Platform branching |
| `selection-active-any?` | Returns `#f` | Editor state |
| `window-visible?` | Returns `#f` | GUI state |
| `get-output-tree` | Returns `""` | Tree output |
| `cpp-string->object` | Identity | C++ interop |
| `object->string` | Identity | C++ interop |
| `get-boolean-preference` | Returns `#f` | Preferences |
| `get-string-preference` | Returns `""` | Preferences |
| `cpp-get-default-library` | Returns `""` | Library paths |
| `get-document-language` | Returns `"english"` | Localization |
| `get-user-language` | Returns `"english"` | Localization |
| `tree->stree` | Identity | Tree conversion |
| `tree->object` | Returns `""` | Tree conversion |
| `select` | noop | Editor selection |
| `tree-search-upwards` | Returns `#f` | Tree search |
| `tm?` | Returns `#f` | TeXmacs tree predicates |
| `tree-atomic?` | Returns `#f` | TeXmacs tree predicates |
| `tree-compound?` | Returns `#f` | TeXmacs tree predicates |
| `tree-in?` | Returns `#f` | TeXmacs tree predicates |
| `tree-label` | Returns `""` | TeXmacs tree accessors |
| `tree-children` | Returns `'()` | TeXmacs tree accessors |
| `compile-interface-spec` | Identity | Module interface compilation |
| `process-use-modules` | eval-based fallback | Module loading |

## Missing Scheme Built-ins (Added as Stubs)

| Function | Guile 1.8 | Note |
|----------|-----------|------|
| `reverse!` | Present | Destructive list reverse |
| `keyword?` | Present | Keyword predicate |
| `caar`, `cdar`, `caaar`, `caadr`, `cadar`, `cdaar`, `cdadr`, `cddar`, `cdddr` | Present | Missing c*r compositions |
| `make-ahash-table`, `ahash-ref`, `ahash-set!`, `ahash-remove!` | TeXmacs | Association hash tables (built on Guile hash tables) |
| `string-replace` | TeXmacs | String substitution |
| `string-append-suffix`, `string-append-prefix` | TeXmacs | String utilities |
| `ca*r`, `ca*adr` | TeXmacs | Recursive car accessors |

## Architecture-Level Gaps

1. **Module scoping during macro expansion** — When `define-module` switches the current module, `define`-d variables go into that module's obarray. But when a macro defined in the module is later expanded, private `define`-d variables (`cur-props`, etc.) aren't visible. Guile 1.8 captures the module's lexical environment in macro transformers.

2. **Special form vs macro duality** — `use-modules` is a hardcoded special form in pscm but a macro in Guile 1.8. Special forms can't be aliased with `define` or passed to `for-each`/`map`. Guile 1.8 implements module operations as macros, making them first-class.

3. **Macro expansion in expression position** — When a macro object (unexpanded) appears in expression position, pscm reports "not supported expression type: macro" instead of expanding it. This blocks nested macro usage like `(for ...)` inside a `define-macro` body.

## Recommendation (Updated 2026-05-05)

**All 45 kernel/ modules load without errors.** The KERNEL-LOADED milestone has been reached.

Three previously-blocking issues were resolved:
1. Module-scoped macro expansion — fixed by module environment chain (commits 741919b, 1ed79fb)
2. `with` macro destructuring — fixed by detecting pair patterns and using `(apply (lambda ...) val)` (41f2ef3)
3. `assert(is_sym(...))` crash in param binding — replaced with proper `eval_error` (41f2ef3)

Tracks A (stability) and B (C API) are also complete:
- A1: Float comparison — already fixed (63b94ad, 49d3cf1)
- A2: GC abort() — already fixed (e9bd053)
- A3: Print buffer 4096-byte limit — fixed (daf62ab)
- B: 6 C API module wrappers — implemented (e99e490, 968ebf7, 05976ba)

**Next step: C3 — load generic/ and remaining modules (~350 files).**
