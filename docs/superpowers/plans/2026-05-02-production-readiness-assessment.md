# Production Readiness Assessment for TeXmacs Compatibility — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit pscm's Guile 1.8 feature coverage, write targeted compatibility tests, and iteratively load real TeXmacs Scheme code to discover the gap between current state and production readiness.

**Architecture:** Three sequential phases. Phase 1 reads pscm source to produce a feature inventory table. Phase 2 uses that inventory to write focused regression tests for the Guile 1.8 features TeXmacs depends on. Phase 3 feeds actual TeXmacs init files to pscm in an iterative fix-and-retry loop.

**Tech Stack:** pscm (C++20 Scheme interpreter), TeXmacs Scheme source (~238K lines), lit test framework

---

## Phase 1: Systematic Feature Audit

### Task 1: Audit Special Forms

**Files:**
- Read: `src/c/eval.cc:257-500`, `src/c/eval.h`, `src/c/define.cc`, `src/c/let.cc`, `src/c/cond.cc`, `src/c/case.cc`, `src/c/do.cc`, `src/c/delay.cc`, `src/c/quasiquote.cc`

- [ ] **Step 1: Extract special form dispatch from eval.cc**

Read `src/c/eval.cc` lines 257-500. The special form dispatch is a chain of `is_sym_val(l->data, "...")` checks. Catalog every handled form.

- [ ] **Step 2: Compare against Guile 1.8 special form list**

Guile 1.8 special forms (from the Guile Reference Manual §5.2):
`define`, `lambda`, `if`, `cond`, `case`, `do`, `let`, `let*`, `letrec`, `and`, `or`, `delay`, `quasiquote`, `quote`, `set!`, `begin`, `define-macro`, `define-module`, `use-modules`, `export`, `re-export`, `define-public`, `call/cc` / `call-with-current-continuation`, `apply`, `map`, `for-each`, `map-in-order`, `call-with-values`, `dynamic-wind`

For each, check whether pscm has a handler. Mark the result in a table row.

- [ ] **Step 3: Check each special form implementation file for completeness**

For each handled form, read the corresponding `.cc` file (e.g., `define.cc` for `define`) and note any partial implementation indicators: TODO comments, limited argument handling, missing edge cases.

- [ ] **Step 4: Compile special form audit table**

Create a markdown table:

```markdown
| Special Form | Status | Implementation | Notes |
|-------------|--------|---------------|-------|
| define | present | define.cc | full R5RS define |
| lambda | present | eval.cc:eval_lambda | rest args, varargs |
| ... | | | |
```

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-05-02-guile-1.8-feature-audit.md
git commit -m "audit: add Guile 1.8 special forms feature audit"
```

### Task 2: Audit Built-in Procedures

**Files:**
- Read: `src/c/init.cc`, `src/c/list.cc`, `src/c/string.cc`, `src/c/vector.cc`, `src/c/hash_table.cc`, `src/c/number.cc`, `src/c/port.cc`, `src/c/char.cc`, `src/c/predicate.cc`, `src/c/eq.cc`, `src/c/alist.cc`, `src/c/function.cc`, `src/c/procedure.cc`, `src/c/values.cc`, `src/c/symbol.cc`, `src/c/environment.cc`, `src/c/load.cc`, `src/c/error.cc`, `src/c/exit.cc`, `src/c/variable.cc`, `src/c/macro.cc`

- [ ] **Step 1: Extract all registered built-in function names**

Run from the build directory (already done in exploration, verify the list):

```bash
grep -rn 'scm_define_function' src/c/ --include='*.cc' | \
  sed 's/.*scm_define_function("\([^"]*\)".*/\1/' | sort
```

- [ ] **Step 2: Categorize by domain**

Group each built-in into a category: list, string, vector, hash-table, numeric, port/io, character, predicate, equality, association-list, function/procedure, values/multiple-values, symbol, environment, load/eval, error/exit, variable, macro, module.

- [ ] **Step 3: Compare against Guile 1.8 built-in list**

For each category, list the Guile 1.8 built-ins (from R5RS + Guile extensions). Mark pscm coverage as present/partial/missing.

Key Guile 1.8 built-ins to check by category:

**List:** `cons`, `car`, `cdr`, `caar`, `cadr`, `cdar`, `cddr`, `caaar`, `caadr`, `cadar`, `caddr`, `cdaar`, `cdadr`, `cddar`, `cdddr`, `null?`, `list?`, `list`, `length`, `append`, `reverse`, `list-tail`, `list-ref`, `list-head`, `last-pair`, `memq`, `memv`, `member`, `set-car!`, `set-cdr!`, `assq`, `assv`, `assoc`, `assq-ref`, `assq-set!`, `assoc-ref`, `assoc-set!`, `assoc-remove!`, `acons`

**String:** `string?`, `make-string`, `string`, `string-length`, `string-ref`, `string-set!`, `string=?`, `string<?`, `string<=?`, `string>?`, `string>=?`, `string-ci=?`, `string-ci<?`, `string-ci<=?`, `string-ci>?`, `string-ci>=?`, `substring`, `string-append`, `string->list`, `list->string`, `string-fill!`

**Vector:** `vector?`, `make-vector`, `vector`, `vector-length`, `vector-ref`, `vector-set!`, `vector->list`, `list->vector`, `vector-fill!`

**Hash table:** `hash-ref`, `hash-set!`, `hash-remove!`, `hashq-ref`, `hashq-set!`, `hashv-ref`, `hashv-set!`, `hash-get-handle`, `hash-create-handle!`, `hash-fold`, `hashq-get-handle`, `hashq-create-handle!`, `hashq-remove!`, `hashv-get-handle`, `hashv-create-handle!`, `hashv-remove!`

**Numeric:** `number?`, `complex?`, `real?`, `rational?`, `integer?`, `exact?`, `inexact?`, `=?`, `<?`, `<=?`, `>?`, `>=?`, `zero?`, `positive?`, `negative?`, `odd?`, `even?`, `max`, `min`, `+`, `-`, `*`, `/`, `abs`, `quotient`, `remainder`, `modulo`, `sqrt`, `expt`, `gcd`, `lcm`

**Ports/IO:** `input-port?`, `output-port?`, `current-input-port`, `current-output-port`, `current-error-port`, `set-current-input-port`, `set-current-output-port`, `set-current-error-port`, `open-input-file`, `open-output-file`, `close-input-port`, `close-output-port`, `open-input-string`, `open-output-string`, `get-output-string`, `read`, `write`, `display`, `newline`

**Character:** `char?`, `char=?`, `char<?`, `char>?`, `char<=?`, `char>=?`, `char-ci=?`, `char-ci<?`, `char-ci>?`, `char-ci<=?`, `char-ci>=?`, `char-alphabetic?`, `char-numeric?`, `char-whitespace?`, `char-upper-case?`, `char-lower-case?`, `char->integer`, `integer->char`, `char-upcase`, `char-downcase`

**Symbol:** `symbol?`, `symbol->string`, `string->symbol`, `gensym`

**Predicates:** `boolean?`, `pair?`, `symbol?`, `number?`, `string?`, `char?`, `vector?`, `procedure?`, `null?`, `eof-object?`

**Equality:** `eq?`, `eqv?`, `equal?`

**Control features:** `apply`, `map`, `for-each`, `force`, `call-with-current-continuation`, `values`, `call-with-values`, `dynamic-wind`

**Other:** `procedure-property`, `set-procedure-property!`, `procedure-name`, `eval`, `load`, `primitive-load`, `error`, `exit`, `defined?`, `module-ref`, `module-bound?`, `module-map`, `module?`, `make-variable`, `variable?`, `variable-ref`, `variable-set!`, `variable-bound?`, `make-undefined-variable`, `resolve-module`, `set-current-module`, `current-module`, `set-module-binder!`, `macroexpand`, `macroexpand-1`, `scm-make-smob-type`, `scm-make-smob`, `scm-smob?`, `scm-smob-data`, `noop`

- [ ] **Step 4: Append built-in audit to the feature audit document**

Add a second markdown table under "## Built-in Procedures".

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-05-02-guile-1.8-feature-audit.md
git commit -m "audit: add Guile 1.8 built-in procedures feature audit"
```

### Task 3: Audit Module System

**Files:**
- Read: `src/c/module.cc` (entire file), `src/c/eval.cc` (module-related special forms: eval_define_module, eval_use_modules, eval_export, eval_re_export, eval_define_public)

- [ ] **Step 1: List all module operations in pscm**

From `src/c/module.cc`, catalog every `scm_*` function related to modules:
- Module creation: `scm_make_module`, `scm_resolve_module`, `scm_c_resolve_module`
- Variable lookup: `scm_module_variable`, `scm_module_lookup`, `scm_c_module_lookup`
- Definition: `scm_module_define`, `scm_c_module_define`
- Export: `scm_module_export`, `scm_ensure_public_interface`
- Use/import: `scm_c_module_use`, `scm_c_use_module`, `scm_module_use!`
- Binder: `scm_set_module_binder_x`, `make_autoload_binder`
- Introspection: `scm_module_map`, `scm_module_kind`
- Registration: `scm_module_system_star_set_x`

- [ ] **Step 2: Compare against Guile 1.8 module API**

Guile 1.8 module API to check:
| Function | pscm equivalent |
|----------|----------------|
| `define-module` | `eval_define_module` |
| `use-modules` | `eval_use_modules` |
| `export` | `eval_export` |
| `re-export` | `eval_re_export` |
| `define-public` | `eval_define_public` |
| `module-use!` | `module-use!` (Scheme) / `scm_module_use!` (C) |
| `module-export!` | check if implemented |
| `module-ref` | `scm_module_variable` wrapper |
| `resolve-module` | `scm_resolve_module` |
| `resolve-interface` | check if implemented |
| `module-map` | `scm_module_map` |
| `module-obarray` | check if exposed |
| `module-public-interface` | `module->public_interface` |
| `module-name` | check if implemented |
| `module-binder` | `module->binder` (check if exposed to Scheme) |
| `set-module-binder!` | present (Scheme built-in) |
| `current-module` | present (Scheme built-in) |
| `set-current-module` | present (Scheme built-in) |
| `%module-public-interface` | present (internal symbol) |

- [ ] **Step 3: Check define-module option coverage**

Review `eval_define_module` option parsing against the design spec list: `#:use-module`, `#:export`, `#:pure`, `#:autoload`, `#:no-backtrace`, `#:renamer`. All of these should be implemented per the recent module system completion work. Verify.

- [ ] **Step 4: Append module audit to feature audit document**

Add a third section "## Module System".

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-05-02-guile-1.8-feature-audit.md
git commit -m "audit: add Guile 1.8 module system feature audit"
```

### Task 4: Audit Macros, Ports, and Reader/Printer

**Files:**
- Read: `src/c/macro.cc`, `src/c/port.cc`, `src/c/read.cc`, `src/c/print.cc`, `src/c/parse.cc`

- [ ] **Step 1: Audit macro system**

Read `src/c/macro.cc`. Check:
- `define-macro` — is the transformer properly invoked? Does expansion happen before eval (as seen in eval.cc:300-323)?
- `macroexpand` / `macroexpand-1` — are these Scheme-level functions present?
- Macro expansion ordering — does pscm expand macros top-down or bottom-up? Guile 1.8 expands top-down (outermost macro first). Check `expand_macros()`.
- `syntax-rules` — is hygienic macro support present at all? (Likely missing; note it.)

- [ ] **Step 2: Audit port system**

Read `src/c/port.cc`. Check:
- File ports: `open-input-file`, `open-output-file`, `close-input-port`, `close-output-port`
- String ports: `open-input-string`, `open-output-string`, `get-output-string`
- Soft ports: Check if implemented (likely not — note as gap)
- Current port management: `current-input-port`, `current-output-port`, `current-error-port`, and their `set!` variants
- Port flags: check for `(current-reader port)` / fluid support for ports

- [ ] **Step 3: Audit reader/printer**

Read `src/c/read.cc` and `src/c/print.cc`. Check:
- `read` — does it handle shared structure (`#n#` / `#n=` notation)?
- Character literals (`#\space`, `#\newline`, etc.)
- String escapes (`\n`, `\t`, `\\`, `\"`, `\xNN`)
- Keyword syntax (`#:key` — recently added per commit `b59fefd`)
- Quotation shorthand (`'x` → `(quote x)`, `` `x`` → `(quasiquote x)`, `,x` → `(unquote x)`, `,@x` → `(unquote-splicing x)`)
- `write` / `display` — do they handle all pscm types correctly?
- `newline`, `write-char`, `read-char` — basic I/O

- [ ] **Step 4: Append to audit document**

Add sections "## Macro System", "## Ports and I/O", "## Reader/Printer".

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-05-02-guile-1.8-feature-audit.md
git commit -m "audit: add macro, port, reader/printer feature audit"
```

### Task 5: Audit C API and Numeric Tower

**Files:**
- Read: `src/c/pscm_api.h`, `src/c/pscm_api.cc`, `src/c/number.cc`

- [ ] **Step 1: Audit C API**

Read `src/c/pscm_api.h`. Catalog every `pscm_*` function and compare against the C API completeness design spec (`docs/superpowers/specs/2026-05-01-c-api-completeness-design.md`). Mark each function from the spec as present/missing:

Phase A (Core): `pscm_init`, `pscm_eval`, `pscm_eval_string`, `pscm_parse`, `pscm_c_define`, `pscm_c_define_gsubr`, `pscm_call_0` through `pscm_call_3`
Phase B (Module): `pscm_c_resolve_module`, `pscm_c_module_lookup`, `pscm_c_module_define`, `pscm_c_use_module`, `pscm_c_current_module`, `pscm_c_set_current_module`
Phase C (Ports): `pscm_current_input_port`, `pscm_current_output_port`, etc.

Read `src/c/pscm_api.cc` for implementation status.

- [ ] **Step 2: Audit numeric tower**

Read `src/c/number.cc`. Check arithmetic operation coverage against R5RS:
- Generic arithmetic: `+`, `-`, `*`, `/`, `abs`, `quotient`, `remainder`, `modulo`
- Comparison: `=`, `<`, `>`, `<=`, `>=`
- Predicates: `number?`, `integer?`, `real?`, `complex?`, `rational?`, `exact?`, `inexact?`, `odd?`, `even?`, `zero?`, `positive?`, `negative?`
- Advanced: `sqrt`, `expt`, `gcd`, `lcm`, `max`, `min`
- Check float precision: is `double` used? Are there float→int coercion rules?
- Check ratio support: `src/c/pscm_types.h` mentions `SCM_RATIO` type. Is it fully implemented?

- [ ] **Step 3: Append to audit document**

Add sections "## C API" and "## Numeric Tower".

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-05-02-guile-1.8-feature-audit.md
git commit -m "audit: add C API and numeric tower feature audit"
```

### Task 6: Audit Remaining Systems and Finalize

**Files:**
- Read: `src/c/continuation.cc`, `src/c/dynwind.cc`, `src/c/throw.cc`, `src/c/throw.h`, `src/c/values.cc`, `src/c/gc.cc`, `src/c/gc.h`, `src/c/smob.cc`, `src/c/smob.h`, `src/c/variable.cc`, `src/c/abort.cc`, `src/c/apply.cc`, `src/c/for_each.cc`, `src/c/map.cc`, `src/c/load.cc`, `src/c/debug.cc`

- [ ] **Step 1: Audit continuations**

Read `src/c/continuation.cc`. Check:
- `call/cc` / `call-with-current-continuation` — does it capture the full continuation state?
- Is module state saved/restored? (Check for `saved_module` field in `SCM_Continuation`)
- Is the wind chain saved/restored?
- Known limitation: pscm uses `setjmp`/`longjmp`. Does this interact correctly with C++ destructors?

- [ ] **Step 2: Audit dynamic-wind**

Read `src/c/dynwind.cc`. Check:
- `dynamic-wind` — is before/after thunk ordering correct?
- Wind chain management in `throw.cc` — are wind guards run during non-local exit?
- Lazy catch handlers in `throw.cc` — are they compatible with Guile 1.8's `lazy-catch`?

- [ ] **Step 3: Audit multiple values**

Read `src/c/values.cc`. Check:
- `values` — can it return zero or multiple values?
- `call-with-values` — does it properly collect multiple values from the producer?
- How are multiple values represented internally? (Check `SCM_Values` type)

- [ ] **Step 4: Audit GC**

Read `src/c/gc.cc` and `src/c/gc.h`. Note current limits and risks:
- `MAX_SEGMENTS = 16`, `MAX_ROOTS = 2048`, `GROWTH_SIZE = 4MB`
- Conservative stack scan: reports as present (implemented), but note the inherent risk of false retention / missed roots
- Mark-sweep: note that it's stop-the-world (no incremental/concurrent GC)
- Note 4 remaining `abort()` calls for heap exhaustion (mmap failure, segment pool full, root limit exceeded)

- [ ] **Step 5: Audit SMOB system**

Read `src/c/smob.cc`. Guile 1.8 SMOBs are user-defined C types callable from Scheme. Check:
- `scm-make-smob-type` — can user code register new SMOB types?
- `scm-make-smob` — can user code create SMOB instances?
- `scm-smob?` / `scm-smob-data` — can user code inspect SMOBs?
- Mark/sweep/free/print hooks for SMOB types? (Check design spec section)

- [ ] **Step 6: Audit debug system**

Read `src/c/debug.cc`. TeXmacs init uses `debug-set!`, `debug-enable`. Check if these are present and functional.

- [ ] **Step 7: Finalize audit document**

Combine all sections into a single `docs/superpowers/specs/2026-05-02-guile-1.8-feature-audit.md`. Add a summary section at the top with counts: X present, Y partial, Z missing. Add a "Top Risks for TeXmacs" section identifying the highest-impact gaps.

- [ ] **Step 8: Commit**

```bash
git add docs/superpowers/specs/2026-05-02-guile-1.8-feature-audit.md
git commit -m "audit: complete Guile 1.8 feature audit with summary and risk assessment"
```

---

## Phase 2: Focused Compatibility Tests

### Task 7: Module System Edge Case Tests

**Files:**
- Create: `test/module/texmacs/nested_module.scm`
- Create: `test/module/texmacs/Output/nested_module.script`
- Create: `test/module/texmacs/re_export_conflict.scm`
- Create: `test/module/texmacs/Output/re_export_conflict.script`
- Create: `test/module/texmacs/module_use_binding.scm`
- Create: `test/module/texmacs/Output/module_use_binding.script`

These tests go beyond the existing module smoke tests. They exercise patterns that the audit identified as partial or that TeXmacs's `inherit-modules` macro relies on.

- [ ] **Step 1: Write nested module test**

Create `test/module/texmacs/nested_module.scm`:

```scheme
;; Test: nested module resolution — modules that use other modules
(use-modules (test nested a))
(display (module-ref (resolve-module '(test nested a)) 'value-a)) (newline)

;; Verify that a module defined with #:use-module sees the used module's exports
(display (module-ref (resolve-module '(test nested b)) 'value-b)) (newline)

;; Verify isolation: module B's internal symbol should NOT be visible
(display (module-bound? (resolve-module '(test nested b)) 'internal-b)) (newline)
```

Create `test/module/texmacs/nested_a.scm`:

```scheme
(define-module (test nested a)
  #:export (value-a internal-a))
(define value-a "a-value")
(define internal-a "internal-a")
```

Create `test/module/texmacs/nested_b.scm`:

```scheme
(define-module (test nested b)
  #:use-module (test nested a)
  #:export (value-b))
(define value-b (string-append "b-uses-" value-a))
(define internal-b "internal-b")
```

Create `test/module/texmacs/Output/nested_module.script`:

```
RUN: %pscm_cc --test %S/nested_module.scm 2>&1 | %filter
CHECK: a-value
CHECK: b-uses-a-value
CHECK: #f
```

Create `test/module/texmacs/nested_a.scm` and `test/module/texmacs/nested_b.scm` as helper modules (no separate lit test — they're loaded by `nested_module.scm` via `%load-path`).

Register these in the CMakeLists.txt for `test/module/` following the existing pattern (e.g., how `module_texmacs_module` test is registered).

- [ ] **Step 2: Write re-export conflict test**

Create `test/module/texmacs/re_export_conflict.scm`:

```scheme
;; Test: re-export when two used modules export the same symbol
;; Guile 1.8 behavior: the FIRST module in the use list wins (or error, depending on config)
;; We mainly care that pscm doesn't crash or silently corrupt state.
(display "Testing re-export...") (newline)
(display (module-ref (resolve-module '(test rexport consumer)) 'shared-name)) (newline)
(display "PASS") (newline)
```

Create helper modules `test/module/texmacs/rexport_a.scm`:

```scheme
(define-module (test rexport a) #:export (shared-name))
(define shared-name "from-a")
```

Create `test/module/texmacs/rexport_b.scm`:

```scheme
(define-module (test rexport b) #:export (shared-name))
(define shared-name "from-b")
```

Create `test/module/texmacs/rexport_consumer.scm`:

```scheme
(define-module (test rexport consumer)
  #:use-module (test rexport a)
  #:use-module (test rexport b))
```

Create `test/module/texmacs/Output/re_export_conflict.script`:

```
RUN: %pscm_cc --test %S/re_export_conflict.scm 2>&1 | %filter
CHECK: PASS
```

- [ ] **Step 3: Write module-use! ordering test**

Create `test/module/texmacs/module_use_binding.scm`:

```scheme
;; Test: module-use! then define should shadow the import
(define-module (test usebind target) #:export (foo))
(define foo "target-foo")

(define-module (test usebind consumer))
(module-use! (resolve-module '(test usebind consumer))
             (resolve-interface '(test usebind target)))
;; Shadow via local define
(define foo "local-foo")
(display foo) (newline)
(display "PASS") (newline)
```

Create `test/module/texmacs/Output/module_use_binding.script`:

```
RUN: %pscm_cc --test %S/module_use_binding.scm 2>&1 | %filter
CHECK: local-foo
CHECK: PASS
```

- [ ] **Step 4: Build and run the new tests**

```bash
cmake --preset pscm-cmake
ninja -C out/build/pscm-cmake
ctest -R module -R "nested_module|re_export_conflict|module_use_binding" --output-on-failure
```

Expected: tests pass or reveal specific gaps.

- [ ] **Step 5: Commit**

```bash
git add test/module/texmacs/
git commit -m "test: add module system edge case tests for TeXmacs compatibility"
```

### Task 8: TeXmacs Macro Compatibility Tests

**Files:**
- Create: `test/module/texmacs/texmacs_macros.scm`
- Create: `test/module/texmacs/Output/texmacs_macros.script`
- Create: `test/module/texmacs/define_public_macro.scm`
- Create: `test/module/texmacs/Output/define_public_macro.script`

- [ ] **Step 1: Write texmacs-module macro test**

Create `test/module/texmacs/texmacs_macros.scm`:

```scheme
;; Test: texmacs-module macro with all options TeXmacs uses
(load "boot.scm")   ;; loads texmacs-user definemodule, provide-public, inherit-modules, texmacs-module macros
(load "m.scm")      ;; defines module (m) with my-identity

;; Test :use option
(texmacs-module (test use)
  (:use (m)))

;; Test :inherit option
(texmacs-module (test inherit)
  (:inherit (m)))

(display (module-ref (resolve-module '(test use)) 'my-identity)) (newline)
(display (module-ref (resolve-module '(test inherit)) 'my-identity)) (newline)
(display "PASS") (newline)
```

Create `test/module/texmacs/Output/texmacs_macros.script`:

```
RUN: %pscm_cc --test %S/texmacs_macros.scm 2>&1 | %filter
CHECK: PASS
```

- [ ] **Step 2: Write define-public-macro test**

Create `test/module/texmacs/define_public_macro.scm`:

```scheme
;; Test: define-public-macro (TeXmacs uses this to export macros)
(define-module (test dpm))
(define-public-macro (my-when test . body)
  `(if ,test (begin ,@body)))

(export my-when)  ;; should also work with define-public-macro
(my-when #t (display "works") (newline))
(display "PASS") (newline)
```

Create `test/module/texmacs/Output/define_public_macro.script`:

```
RUN: %pscm_cc --test %S/define_public_macro.scm 2>&1 | %filter
CHECK: PASS
```

- [ ] **Step 3: Build and run tests**

```bash
ninja -C out/build/pscm-cmake
ctest -R texmacs_macros\|define_public_macro --output-on-failure
```

- [ ] **Step 4: Commit**

```bash
git add test/module/texmacs/
git commit -m "test: add TeXmacs macro compatibility tests"
```

### Task 9: Port and I/O Tests

**Files:**
- Create: `test/module/texmacs/string_ports.scm`
- Create: `test/module/texmacs/Output/string_ports.script`
- Create: `test/module/texmacs/file_ports.scm`
- Create: `test/module/texmacs/Output/file_ports.script`

These templates are approximate — adjust based on Phase 1 audit findings for port support.

- [ ] **Step 1: Write string port test**

Create `test/module/texmacs/string_ports.scm`:

```scheme
;; Test: string port round-trip (TeXmacs uses string ports for document processing)
(define s (open-output-string))
(display "hello" s)
(display " world" s)
(define result (get-output-string s))
(display result) (newline)
(display (string=? result "hello world")) (newline)

;; Test: input string port
(define in (open-input-string "scheme data"))
(display (read in)) (newline)
(display (read-char in)) (newline)
(display "PASS") (newline)
```

Create `test/module/texmacs/Output/string_ports.script`:

```
RUN: %pscm_cc --test %S/string_ports.scm 2>&1 | %filter
CHECK: hello world
CHECK: #t
CHECK: scheme
CHECK: PASS
```

- [ ] **Step 2: Write file port test**

Create `test/module/texmacs/file_ports.scm`:

```scheme
;; Test: file port basics
(define tmpfile "test_port_tmp.txt")
(define out (open-output-file tmpfile))
(display "test content\n" out)
(close-output-port out)
(define in (open-input-file tmpfile))
(display (read-line in)) (newline)
(close-input-port in)
(display "PASS") (newline)
```

Create `test/module/texmacs/Output/file_ports.script`:

```
RUN: %pscm_cc --test %S/file_ports.scm 2>&1 | %filter
CHECK: PASS
```

Note: `read-line` may not exist in pscm. If the audit shows it's missing, adjust this test to use `read` + `read-char` instead, or skip this test and file an issue.

- [ ] **Step 3: Build and run tests**

```bash
ninja -C out/build/pscm-cmake
ctest -R string_ports\|file_ports --output-on-failure
```

- [ ] **Step 4: Commit**

```bash
git add test/module/texmacs/
git commit -m "test: add port and I/O compatibility tests"
```

### Task 10: Continuation + Dynamic-wind + Module Interaction Tests

**Files:**
- Create: `test/module/texmacs/cont_module.scm`
- Create: `test/module/texmacs/Output/cont_module.script`
- Create: `test/module/texmacs/dynwind_module.scm`
- Create: `test/module/texmacs/Output/dynwind_module.script`

- [ ] **Step 1: Write continuation + module test**

Create `test/module/texmacs/cont_module.scm`:

```scheme
;; Test: call/cc across module boundaries
;; Continuation capture in one module, invoke in another
(define-module (test cont a) #:export (capture))
(define saved #f)
(define (capture)
  (call/cc (lambda (k) (set! saved k) 'captured)))
(define (get-saved) saved)

(define-module (test cont b) #:use-module (test cont a))
(display (capture)) (newline)  ;; should print 'captured
(display "PASS") (newline)
```

Create `test/module/texmacs/Output/cont_module.script`:

```
RUN: %pscm_cc --test %S/cont_module.scm 2>&1 | %filter
CHECK: captured
CHECK: PASS
```

- [ ] **Step 2: Write dynamic-wind + module test**

Create `test/module/texmacs/dynwind_module.scm`:

```scheme
;; Test: dynamic-wind with module-local state
(define-module (test wind m) #:export (test-wind))
(define wind-log '())
(define (test-wind)
  (dynamic-wind
    (lambda () (set! wind-log (cons 'before wind-log)))
    (lambda () (set! wind-log (cons 'during wind-log)) 'thunk-result)
    (lambda () (set! wind-log (cons 'after wind-log))))
  wind-log)

(display (test-wind)) (newline)
(display "PASS") (newline)
```

Create `test/module/texmacs/Output/dynwind_module.script`:

```
RUN: %pscm_cc --test %S/dynwind_module.scm 2>&1 | %filter
CHECK: PASS
```

- [ ] **Step 3: Build and run tests**

```bash
ninja -C out/build/pscm-cmake
ctest -R cont_module\|dynwind_module --output-on-failure
```

- [ ] **Step 4: Commit**

```bash
git add test/module/texmacs/
git commit -m "test: add continuation and dynamic-wind module interaction tests"
```

---

## Phase 3: Incremental Loading

### Task 11: Set Up Incremental Loading Scaffold

**Files:**
- Create: `test/module/texmacs/load_texmacs_init.scm` — driver script
- Modify: `test/module/CMakeLists.txt` — register new test

- [ ] **Step 1: Create the incremental loading driver**

Create `test/module/texmacs/load_texmacs_init.scm`:

```scheme
;; Incremental loading test for TeXmacs init sequence.
;; This is the harness — it loads init-texmacs.scm and reports
;; success or the first error encountered.

;; Set up load path to find TeXmacs progs
(set! %load-path (cons "/Users/pikachu/pr/texmacs/TeXmacs/progs" %load-path))
(set! %load-path (cons "/Users/pikachu/pr/texmacs/TeXmacs/fonts" %load-path))

;; TeXmacs init needs a C++-side module for preferences/dialect detection.
;; Define stubs for required C++ functions that init-texmacs.scm calls:
(define (cpp-get-preference key default) default)
(define (os-mingw?) #f)
(define (os-macos?) #t)  ;; we are on macOS in this test
(define (texmacs-time) 0)
(define (tm-interactive) #f)
(define (scheme-dialect) "guile-a")
(define (supports-email?) #f)
(define (symbol-property sym key) #f)
(define (set-symbol-property! sym key val) (noop))
(define (source-property form key)
  (if (eq? key 'line) 0
      (if (eq? key 'column) 0
          (if (eq? key 'filename) "unknown" #f))))
(define (debug-set! key val) (noop))
(define (debug-enable . args) (noop))
(define module-export! export)
(define (with-fluids fluids thunk) (thunk))

;; Try loading the init file
(catch #t
  (lambda ()
    (load "/Users/pikachu/pr/texmacs/TeXmacs/progs/init-texmacs.scm")
    (display "INIT-LOADED") (newline))
  (lambda (key . args)
    (display "ERROR: ") (display key) (newline)
    (display "ARGS: ") (display args) (newline)))
```

Create `test/module/texmacs/Output/load_texmacs_init.script`:

```
RUN: %pscm_cc --test %S/load_texmacs_init.scm 2>&1 | %filter
CHECK: INIT-LOADED
```

The first run is expected to FAIL (the `CHECK: INIT-LOADED` line won't match). We iterate from here.

- [ ] **Step 2: Register the test in CMakeLists.txt**

Add to `test/module/CMakeLists.txt` (check exact path and follow existing patterns for `module_texmacs_module` test):

```cmake
# Incremental TeXmacs init loading test
add_test(NAME texmacs_init_loading
  COMMAND ${CMAKE_BINARY_DIR}/pscm_cc --test
    ${CMAKE_CURRENT_SOURCE_DIR}/texmacs/load_texmacs_init.scm)
set_tests_properties(texmacs_init_loading PROPERTIES
  PASS_REGULAR_EXPRESSION "INIT-LOADED")
```

- [ ] **Step 3: Build and run (expect first failure)**

```bash
ninja -C out/build/pscm-cmake
ctest -R texmacs_init_loading --output-on-failure
```

Expected: FAIL. The error message tells us what broke first — missing function, syntax error, macro expansion failure, or GC crash. This failure is the starting point for Task 12.

- [ ] **Step 4: Commit**

```bash
git add test/module/texmacs/load_texmacs_init.scm test/module/texmacs/Output/load_texmacs_init.script
git add test/module/CMakeLists.txt
git commit -m "test: add incremental TeXmacs init loading scaffold"
```

### Task 12: First Incremental Fix Cycle


**Files:**
- Modify: `test/module/texmacs/load_texmacs_init.scm` — update stub list
- Potentially modify: pscm source files — depending on error

This task is the template for the iterative fix loop. It runs once per blocking error discovered.

- [ ] **Step 1: Run the loading test and capture the error**

```bash
ctest -R texmacs_init_loading --output-on-failure 2>&1 | tee /tmp/texmacs_error_1.txt
```

- [ ] **Step 2: Classify the error**

Read the error output. Classify into one of:

| Category | Example | Action |
|----------|---------|--------|
| Missing C++ stub | `unbound variable: cpp-get-preference` | Add stub to `load_texmacs_init.scm` |
| Missing Scheme built-in | `unbound variable: read-line` | File issue; add stub or implement |
| Macro expansion error | `quasiquote: bad syntax` | Investigate in pscm; fix or file issue |
| GC crash/abort | `FATAL: gc heap mmap failed` | Fix in pscm GC; this is blocking |
| Semantic mismatch | Wrong value returned | Investigate; fix or file issue |

- [ ] **Step 3: Apply the fix**

For missing C++ stubs: add a `(define ...)` to the stub section in `load_texmacs_init.scm`.

For missing Scheme built-ins that TeXmacs needs: if trivial, implement in pscm; otherwise, add a Scheme-level workaround and file an issue.

For macro errors: investigate the expansion behavior. Check whether Guile 1.8 expands differently.

For GC crashes: fix the GC defect in pscm. This is always blocking.

- [ ] **Step 4: Re-run the test**

```bash
ninja -C out/build/pscm-cmake
ctest -R texmacs_init_loading --output-on-failure
```

- [ ] **Step 5: Commit**

Commit message describes what was fixed:

```bash
git add <modified files>
git commit -m "fix: <specific error> in TeXmacs init loading"
```

- [ ] **Step 6: Repeat Steps 1-5 until the test passes**

The cycle continues: run → classify → fix → commit → run again. Each Task 13, 14, ... is one iteration of this loop until `init-texmacs.scm` loads without uncaught errors.

### Task 13-N: Iterative Fix Cycles

**Pattern:** Each subsequent task repeats Task 12's cycle for the next blocking error. Continue until `CHECK: INIT-LOADED` passes.

Expected total iterations: 5-20 cycles depending on gap severity. The most common early failures will be missing C++ stubs (TeXmacs registers many C++ functions via `scm_c_define`). Once the C++ stub layer is complete, the next layer of failures will be macro expansion mismatches and missing Scheme built-ins.

**Critical stopping conditions** (these indicate deeper issues that need a separate spec):
- GC crash that can't be fixed with a small patch
- Macro hygiene gap that requires a macro system rewrite
- More than 3 missing R5RS built-ins per category

### Task last: Document Gap Inventory

**Files:**
- Modify: `docs/superpowers/specs/2026-05-02-guile-1.8-feature-audit.md` — update with Phase 3 findings
- Create: `docs/superpowers/specs/2026-05-02-texmacs-gap-inventory.md`

- [ ] **Step 1: Summarize Phase 3 findings**

Create `docs/superpowers/specs/2026-05-02-texmacs-gap-inventory.md`:

```markdown
# TeXmacs Compatibility Gap Inventory

**Date**: 2026-05-02
**Based on**: Incremental loading of TeXmacs init-texmacs.scm through pscm

## Loading Status

- [ ] init-texmacs.scm loads without errors
- [ ] progs/kernel/ modules resolve
- [ ] progs/utils/ modules resolve
- [ ] progs/generic/ modules resolve

## Blocking Gaps (fixed during Phase 3)

| # | Gap | Error | Fix |
|---|-----|-------|-----|
| 1 | ... | ... | ... |

## Remaining Gaps (filed as issues)

| # | Gap | Severity | Issue |
|---|-----|----------|-------|
| 1 | ... | blocking/non-blocking | #N |

## TeXmacs-Specific C++ Functions Needing Stubs

(List of all C++ functions that TeXmacs registers via scm_c_define
that we stubbed in load_texmacs_init.scm)

## Recommendation

[Can pscm go to production for TeXmacs? Yes / Not yet / With caveats]
```

- [ ] **Step 2: Update feature audit with Phase 3 corrections**

Review the Phase 1 audit tables. Phase 3 may reveal features that the audit marked "present" but that behave differently from Guile 1.8 in practice. Update those entries.

- [ ] **Step 3: Final commit**

```bash
git add docs/superpowers/specs/
git commit -m "docs: TeXmacs compatibility gap inventory and final assessment"
```

---

## Summary

| Phase | Tasks | Output |
|-------|-------|--------|
| 1: Audit | 1-6 | `docs/superpowers/specs/2026-05-02-guile-1.8-feature-audit.md` |
| 2: Tests | 7-10 | New `.scm` tests in `test/module/texmacs/` |
| 3: Loading | 11-N | Iterative fixes + gap inventory |

Total estimated cycles for Phase 3: 5-20 iterations depending on gap severity.
