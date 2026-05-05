# Kernel Loading Gap Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 4 issues blocking Phase 1 kernel loading: `stack` not found (stub), `texmacs-module` not found (auto-fixed), `'|'` parser gap (add `|` to symbol chars), `kernel-base` scope (harness cleanup).

**Architecture:** Each fix is independent. Tasks 1-3 modify single files. Task 4 is a test harness cleanup. Order matters only for verification: Task 1 unblocks Task 2 automatically.

**Tech Stack:** pscm (C++20 Scheme interpreter), CTest

---

## File Structure

| File | Responsibility |
|------|---------------|
| `test/module/texmacs/load_kernel.scm` | Fix stub for `debug-set!`, remove broken save/restore, move `kernel-base` |
| `src/c/parse.cc:810` | Add `|` to symbol character set for Guile 1.8 compatibility |

---

### Task 1: Fix `debug-set!` stub — use `define-macro`

**Files:**
- Modify: `test/module/texmacs/load_kernel.scm`

In Guile 1.8, `debug-set!` is a special form that does NOT evaluate its first argument — `(debug-set! stack 2000000)` works even though `stack` is not a variable. pscm's stub `(define (debug-set! key val) (noop))` is a regular function that evaluates both arguments, causing `stack` to be looked up as a variable.

- [ ] **Step 1: Read the current stub line**

```bash
grep -n 'debug-set!' /Users/pikachu/pr/pscm/test/module/texmacs/load_kernel.scm
```

- [ ] **Step 2: Replace `define` with `define-macro`**

Change:
```scheme
(define (debug-set! key val) (noop))
```
to:
```scheme
(define-macro (debug-set! key val) '(noop))
```

This matches Guile 1.8 semantics: arguments are quoted, not evaluated.

- [ ] **Step 3: Also fix the separate stub in `load_texmacs_init.scm`**

```bash
grep -n 'debug-set!' /Users/pikachu/pr/pscm/test/module/texmacs/load_texmacs_init.scm
```

Apply the same change there.

- [ ] **Step 4: Build and run load_kernel test**

```bash
cd /Users/pikachu/pr/pscm && ninja -C out/build/pscm-cmake && ctest --test-dir out/build/pscm-cmake -R load_kernel --output-on-failure
```

Expected: The `symbol 'stack' not found` error in init-texmacs.scm should be gone. `texmacs-module` errors should also disappear because boot.scm now loads.

- [ ] **Step 5: Full regression**

```bash
ctest --test-dir out/build/pscm-cmake --output-on-failure && ninja -C out/build/pscm-cmake check-base check-cont check-macro check-core
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add test/module/texmacs/load_kernel.scm test/module/texmacs/load_texmacs_init.scm
git commit -m "fix: change debug-set! stub from define to define-macro for Guile 1.8 alignment"
```

---

### Task 2: Add `|` to symbol character set in parser

**Files:**
- Modify: `src/c/parse.cc:810`

TeXmacs menu files use `'|` — quoting the bare symbol `|`. Guile 1.8 accepts `|` as a valid symbol character. pscm's `parse_symbol` allows only `!$%&*+-./:<=>?@^_~` as special characters; `|` is missing.

- [ ] **Step 1: Add `|` to the symbol character set**

In `src/c/parse.cc` line 810, change:
```cpp
         strchr("!$%&*+-./:<=>?@^_~", *p->pos) != nullptr)) {
```
to:
```cpp
         strchr("!$%&*+-./:<=>?@^_~|", *p->pos) != nullptr)) {
```

- [ ] **Step 2: Build and run load_kernel test**

```bash
cd /Users/pikachu/pr/pscm && ninja -C out/build/pscm-cmake && ./out/build/pscm-cmake/pscm_cc --test test/module/texmacs/load_kernel.scm 2>&1 | grep -c "parse error.*'|'"
```

Expected: `0` — no more `|` parse errors.

- [ ] **Step 3: Write a quick verification test**

```bash
echo "(display '|) (newline)" > /tmp/test_pipe_sym.scm && ./out/build/pscm-cmake/pscm_cc --test /tmp/test_pipe_sym.scm 2>&1
```

Expected: prints `|` (the symbol) and exits 0.

- [ ] **Step 4: Full regression**

```bash
ctest --test-dir out/build/pscm-cmake --output-on-failure && ninja -C out/build/pscm-cmake check-base check-cont check-macro check-core
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/c/parse.cc
git commit -m "fix: add pipe character to symbol character set for Guile 1.8 compatibility"
```

---

### Task 3: Fix `kernel-base` scope in test harness

**Files:**
- Modify: `test/module/texmacs/load_kernel.scm`

The `(set-current-module #f)` / restore dance never worked (both pscm and Guile 1.8 reject `#f`). Remove it and define `kernel-base` in the current module where it's visible to the try-load loop.

- [ ] **Step 1: Read current save/restore lines**

```bash
grep -n 'pscm-saved-module\|set-current-module\|kernel-base' /Users/pikachu/pr/pscm/test/module/texmacs/load_kernel.scm
```

- [ ] **Step 2: Remove save/restore, move kernel-base**

Remove lines 11-12:
```scheme
(define pscm-saved-module (current-module))
(set-current-module #f)
```

Remove line 221:
```scheme
(set-current-module pscm-saved-module)
```

Keep `kernel-base` at line 237 but move it before the try-load loop starts — actually it's already at line 237 which is before the `for-each` at line 261. It just needs to be visible. Since we removed the broken save/restore, all defines go into the same module, so `kernel-base` should now be visible during the for-each.

- [ ] **Step 3: Build and verify**

```bash
cd /Users/pikachu/pr/pscm && ninja -C out/build/pscm-cmake && ./out/build/pscm-cmake/pscm_cc --test test/module/texmacs/load_kernel.scm 2>&1 | grep "kernel-base"
```

Expected: no `kernel-base` errors.

- [ ] **Step 4: Full regression**

```bash
ctest --test-dir out/build/pscm-cmake --output-on-failure && ninja -C out/build/pscm-cmake check-base check-cont check-macro check-core
```

- [ ] **Step 5: Commit**

```bash
git add test/module/texmacs/load_kernel.scm
git commit -m "fix: remove broken set-current-module save/restore from kernel loading driver"
```

---

### Task 4: Verify all kernel files load, update gap inventory

- [ ] **Step 1: Run load_kernel and capture results**

```bash
cd /Users/pikachu/pr/pscm && ./out/build/pscm-cmake/pscm_cc --test test/module/texmacs/load_kernel.scm > /tmp/load_kernel_final.txt 2>&1
```

- [ ] **Step 2: Count remaining errors by category**

```bash
grep -c "ERROR:" /tmp/load_kernel_final.txt
grep -c "parse error" /tmp/load_kernel_final.txt
grep -c "texmacs-module" /tmp/load_kernel_final.txt
grep -c "tm-define" /tmp/load_kernel_final.txt
grep -c "stack" /tmp/load_kernel_final.txt
grep -c "kernel-base" /tmp/load_kernel_final.txt
```

Expected: all 6 counts = 0.

- [ ] **Step 3: Count OK files**

```bash
grep -c "OK" /tmp/load_kernel_final.txt
```

This tells us how many of the 20 extra kernel files loaded successfully.

- [ ] **Step 4: Update gap inventory**

Update `docs/superpowers/specs/2026-05-02-texmacs-gap-inventory.md`:
- Mark `texmacs-module` and `tm-define` issues as resolved
- Note `|` parser fix
- List any new gaps discovered

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-05-02-texmacs-gap-inventory.md
git commit -m "docs: update TeXmacs gap inventory after kernel loading fix cycle"
```

---

## Summary

| Task | Fix | File |
|------|-----|------|
| 1 | `debug-set!` → define-macro | `load_kernel.scm`, `load_texmacs_init.scm` |
| 2 | Add `|` to symbol chars | `parse.cc:810` |
| 3 | Remove broken save/restore | `load_kernel.scm` |
| 4 | Verify + update docs | gap inventory |
