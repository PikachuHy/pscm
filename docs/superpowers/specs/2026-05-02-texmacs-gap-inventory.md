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
- [/] Loads kernel/library/* (base, list via inherit-modules)
- [/] Loads kernel/regexp/* (match, select via inherit-modules)
- [/] Loads kernel/logic/* (rules, query, data via inherit-modules)
- [/] Loads kernel/texmacs/tm-define.scm (blocked at macro expansion)

## Blocking Issues Found

| # | Issue | Location | Category | Impact |
|---|-------|----------|----------|--------|
| 1 | Special forms not aliaseable as values | boot.scm:100 `(define import-from use-modules)` | Module system | Prevented boot.scm guile-a path; worked around with scheme-dialect change |
| 2 | Module-scoped privates invisible to macro expansion | tm-define.scm:127-168 (`cur-props`, `cur-props-table`, etc.) | Macro/module | Blocked tm-define macro loading; private defines inside a module not found during expansion |
| 3 | `for` macro in expression position: "not supported expression type: macro" | tm-define.scm:163 | Macro system | Hard block — macro expander not handling nested macro in expression context |
| 4 | Conservative GC `abort()` on heap exhaustion | gc.cc:175,181,303,767 | GC | Risk under heavy loading |

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

## Recommendation

**Not yet production-ready for TeXmacs.** Three blocking issues prevent the TeXmacs init chain from completing:

1. **Module-scoped macro expansion** (blocking) — tm-define.scm uses private module variables during macro expansion. Requires pscm to capture the defining module's environment in macro transformers.
2. **Macro-in-expression-position** (blocking) — The `for` loop macro inside `define-macro` bodies fails. Requires fixing pscm's evaluator to expand macros found in expression position.
3. **Special form aliasability** (workaround exists) — `(define x special-form)` doesn't work. The `scheme-dialect` workaround switches to a `define-macro` path.

Once these three are resolved, the remaining gap is the ~25 C++ stubs that need real implementations (or a strategy for TeXmacs to load without them).

The GC heap limit risk (MAX_SEGMENTS=16, ~64MB) and numeric comparison bugs (<, >, <=, >= truncating doubles) are not currently blocking the init chain but would cause issues under production load.
