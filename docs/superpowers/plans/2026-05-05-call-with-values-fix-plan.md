# call-with-values: Fix Double-Evaluation of Already-Evaluated Values

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `call-with-values` passing already-evaluated values to `apply_procedure` which re-evaluates them, causing symbols to be looked up incorrectly in the environment.

**Architecture:** One-line fix in `values.cc` — use `apply_procedure_with_values` (which treats arguments as already-evaluated) instead of `apply_procedure` (which calls `eval_with_env` on each argument).

**Tech Stack:** pscm (C++20 Scheme interpreter), CTest

---

### Task 1: Fix the bug

**Files:**
- Modify: `src/c/values.cc:77`

**Root cause:** `call-with-values` at `values.cc:77` calls `apply_procedure(env, proc, values_list)`. `apply_procedure` evaluates each argument via `eval_with_env(env, args->data)` (procedure.cc:169). But `values_list` contains ALREADY-EVALUATED values (returned by `scm_c_values`). This causes:
- Symbol values to be looked up in the wrong environment
- Empty lists to pass through (they're self-evaluating)
- Boolean values to pass through (self-evaluating)

The function `apply_procedure_with_values` (procedure.cc:6) exists specifically for this case — it treats `args->data` as already evaluated.

- [ ] **Step 1: Change the call**

In `src/c/values.cc` line 77, change:
```cpp
return apply_procedure(env, proc, values_list);
```
to:
```cpp
return apply_procedure_with_values(env, proc, values_list);
```

- [ ] **Step 2: Build and run the reproduction test**

```bash
cd /Users/pikachu/pr/pscm && ninja -C out/build/pscm-cmake
cat <<'EOF' > /tmp/test_cwv_fix.scm
(call-with-values (lambda () (values '() 'not-bound-sym))
  (lambda (a b)
    (display "a=") (write a) (newline)
    (display "b=") (write b) (newline)
    (display "PASS") (newline)))
EOF
./out/build/pscm-cmake/pscm_cc --test /tmp/test_cwv_fix.scm 2>&1
```

Expected: prints `a=()`, `b=not-bound-sym`, `PASS`. No "symbol not found" error.

- [ ] **Step 3: Run full regression**

```bash
cd /Users/pikachu/pr/pscm && ctest --test-dir out/build/pscm-cmake --output-on-failure && ninja -C out/build/pscm-cmake check-base check-cont check-macro check-core
```

Expected: all tests pass (except possibly load_kernel, which has other issues).

- [ ] **Step 4: Run load_kernel test**

```bash
./out/build/pscm-cmake/pscm_cc --test test/module/texmacs/load_kernel.scm 2>&1 | grep -E 'INIT-ERROR|quasiquote'
```

Expected: quasiquote error should be GONE. 

- [ ] **Step 5: Commit**

```bash
git add src/c/values.cc
git commit -m "fix: use apply_procedure_with_values in call-with-values to prevent double-evaluation"
```
