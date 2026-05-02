# Eliminate exit(1) from Recoverable Error Paths — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** All recoverable Scheme errors return NULL from the public C API instead of killing the host process.

**Architecture:** Wrap `pscm_eval`/`pscm_eval_string`/`pscm_parse` in `scm_c_catch` with catch-all tag `#t`. Convert six exit(1) error sites to use `eval_error` which already calls `scm_throw` → longjmps to the catch handler. The handler stores error details in globals and returns NULL.

**Tech Stack:** C++20, existing pscm codebase

---

### Task 1: Add error storage globals and retrieval API

**Files:**
- Modify: `src/c/pscm_api.cc:10` (add globals)
- Modify: `src/c/pscm_api.h:54-56` (declare new functions)

- [ ] **Step 1: Add error storage globals and new API functions**

Add after the existing `g_error_handler` global in `src/c/pscm_api.cc` line 10:

```cpp
// Error handler callback
static pscm_error_handler_t g_error_handler = nullptr;

// Last error storage (for retrieval after pscm_eval returns NULL).
// Non-static — the catch-all handler in error.cc writes to these.
char *g_last_error_message = nullptr;
char *g_last_error_key = nullptr;
```

Add at the end of `src/c/pscm_api.cc` (before the file ends):

```cpp
const char *pscm_get_last_error_key(void) {
  return g_last_error_key;
}

const char *pscm_get_last_error_message(void) {
  return g_last_error_message;
}
```

Add to `src/c/pscm_api.h` after line 53 (the `pscm_set_error_handler` line):

```c
// Error retrieval (after pscm_eval/pscm_parse returns NULL)
const char *pscm_get_last_error_key(void);
const char *pscm_get_last_error_message(void);
```

- [ ] **Step 2: Build to verify compilation**

```bash
ninja -C out/build/pscm-cmake pscm_cc
```

- [ ] **Step 3: Commit**

```bash
git add src/c/pscm_api.cc src/c/pscm_api.h
git commit -m "feat: add error storage globals and retrieval API"
```

---

### Task 2: Add catch-all error handler helper

**Files:**
- Modify: `src/c/error.cc` (add handler helper)
- Modify: `src/c/error.h` (declare helper)

- [ ] **Step 1: Add `scm_api_catch_handler` function to error.cc**

The catch-all handler that stores error info and returns NULL. Add after the `eval_error` function in `src/c/error.cc` (after line 245):

```cpp
// Catch-all handler for public API boundary.
// Stores error details in globals so the host can retrieve them after
// pscm_eval/pscm_parse returns NULL.
SCM *scm_api_catch_handler(void *data, SCM *tag, SCM *args) {
  (void)data;

  // Free previous error info
  delete[] g_last_error_message;
  delete[] g_last_error_key;
  g_last_error_message = nullptr;
  g_last_error_key = nullptr;

  // Store error tag as a C string
  if (tag && is_sym(tag)) {
    const char *name = cast<SCM_Symbol>(tag)->data;
    g_last_error_key = new char[strlen(name) + 1];
    strcpy(g_last_error_key, name);
  }

  // Extract message from args (Guile convention: args is (msg ...))
  if (args && is_pair(args)) {
    SCM_List *args_list = cast<SCM_List>(args);
    if (args_list->data && is_str(args_list->data)) {
      SCM_String *s = cast<SCM_String>(args_list->data);
      g_last_error_message = new char[s->len + 1];
      memcpy(g_last_error_message, s->data, s->len);
      g_last_error_message[s->len] = '\0';
    }
  }

  return nullptr;
}
```

This function needs access to `g_last_error_message` and `g_last_error_key`. Declare them as extern in error.cc:

Add at the top of `src/c/error.cc` after the includes:

```cpp
extern char *g_last_error_message;
extern char *g_last_error_key;
```

- [ ] **Step 2: Declare in error.h**

Add to `src/c/error.h` before the last line:

```cpp
// Catch-all handler for wrapping public API entry points in scm_c_catch.
// Stores error details for retrieval via pscm_get_last_error_*().
SCM *scm_api_catch_handler(void *data, SCM *tag, SCM *args);
```

- [ ] **Step 3: Build to verify compilation**

```bash
ninja -C out/build/pscm-cmake pscm_cc
```

- [ ] **Step 4: Commit**

```bash
git add src/c/error.cc src/c/error.h
git commit -m "feat: add catch-all error handler for API boundary"
```

---

### Task 3: Wrap pscm_eval, pscm_eval_string, pscm_parse in catch-all

**Files:**
- Modify: `src/c/pscm_api.cc:23-74`

- [ ] **Step 1: Wrap pscm_eval**

Replace the body of `pscm_eval` in `src/c/pscm_api.cc`:

```cpp
// Evaluate an AST node
SCM *pscm_eval(SCM *ast) {
  if (!ast) {
    return nullptr;
  }
  if (!cont_base) {
    long stack_base;
    cont_base = &stack_base;
  }
  return scm_c_catch(
      scm_bool_true(),  // catch-all: catches any throw key
      [](void *data) -> SCM * {
        return eval_with_env(&g_env, (SCM *)data);
      },
      (void *)ast,
      scm_api_catch_handler,
      nullptr);
}
```

- [ ] **Step 2: Wrap pscm_eval_string**

Replace the body of `pscm_eval_string`:

```cpp
SCM *pscm_eval_string(const char *code) {
  if (!code) {
    return nullptr;
  }
  return scm_c_catch(
      scm_bool_true(),
      [](void *data) -> SCM * {
        const char *c = (const char *)data;
        SCM *ast = pscm_parse(c);
        if (!ast) {
          return nullptr;
        }
        return pscm_eval(ast);
      },
      (void *)code,
      scm_api_catch_handler,
      nullptr);
}
```

Note: `pscm_parse` is called inside the catch body, so parse errors get caught.

- [ ] **Step 3: Wrap pscm_parse**

Replace the body of `pscm_parse`:

```cpp
SCM *pscm_parse(const char *code) {
  if (!code) {
    return nullptr;
  }
  return scm_c_catch(
      scm_bool_true(),
      [](void *data) -> SCM * {
        return parse((const char *)data);
      },
      (void *)code,
      scm_api_catch_handler,
      nullptr);
}
```

- [ ] **Step 4: Build to verify compilation**

```bash
ninja -C out/build/pscm-cmake pscm_cc
```

- [ ] **Step 5: Quick smoke test — valid code still works**

```bash
out/build/pscm-cmake/pscm_cc --test test/r4rs/r4rs_mini.scm
```
Expected: "Passed all tests"

- [ ] **Step 6: Quick smoke test — error code returns cleanly**

```bash
echo '(define x)' | out/build/pscm-cmake/pscm_cc --test /dev/stdin 2>&1; echo "exit code: $?"
```
Expected: process does NOT crash (may print error to stderr), exit code 0

- [ ] **Step 7: Commit**

```bash
git add src/c/pscm_api.cc
git commit -m "feat: wrap public API in scm_c_catch to prevent exit(1) on Scheme errors"
```

---

### Task 4: Convert parse.cc exit(1) to eval_error

**Files:**
- Modify: `src/c/parse.cc:146-256`

- [ ] **Step 1: Change parse_error to use eval_error**

In `src/c/parse.cc`, add the include at the top:

```cpp
#include "error.h"
```

Replace the body of `parse_error` (line 146-256). The function currently has extensive error printing followed by `exit(1)`. Replace the last two lines:

Old:
```cpp
  fflush(stderr);
  exit(1);
}
```

New:
```cpp
  fflush(stderr);
  eval_error("Parse error: %s", msg);
}
```

Note: `parse_error` currently is NOT marked `[[noreturn]]`, so no attribute change needed. The function implicitly does not return because `eval_error` calls `scm_throw` which longjmps (or aborts in the no-catch fallback).

- [ ] **Step 2: Build**

```bash
ninja -C out/build/pscm-cmake pscm_cc
```

- [ ] **Step 3: Test parse error does not crash**

```bash
out/build/pscm-cmake/pscm_cc --test /dev/stdin <<< "(invalid" 2>&1; echo "exit=$?"
```
Expected: process prints error, does not exit with non-zero from crash

- [ ] **Step 4: Commit**

```bash
git add src/c/parse.cc
git commit -m "fix: replace exit(1) with eval_error in parse_error"
```

---

### Task 5: Convert eval.h report_arg_mismatch exit(1) to eval_error

**Files:**
- Modify: `src/c/eval.h:92-158`

- [ ] **Step 1: Replace exit(1) with eval_error**

In `src/c/eval.h`, the `report_arg_mismatch` function currently prints extensive error info and calls `exit(1)`. Replace:

Old (line 155-158):
```cpp
  fflush(stderr);
  
  exit(1);
}
```

New:
```cpp
  fflush(stderr);
  
  eval_error("Wrong number of arguments");
}
```

Also update the function signature — remove `[[noreturn]]` since `eval_error` may return in the no-catch fallback (defense in depth):

Old:
```cpp
[[noreturn]] inline void report_arg_mismatch(SCM_List *expected, SCM_List *got, 
                                               const char *call_type = nullptr, 
                                               SCM *original_call = nullptr,
                                               SCM_Symbol *name = nullptr) {
```

New:
```cpp
inline void report_arg_mismatch(SCM_List *expected, SCM_List *got, 
                                const char *call_type = nullptr, 
                                SCM *original_call = nullptr,
                                SCM_Symbol *name = nullptr) {
```

- [ ] **Step 2: Build**

```bash
ninja -C out/build/pscm-cmake pscm_cc
```

- [ ] **Step 3: Test wrong-arg error does not crash**

```bash
out/build/pscm-cmake/pscm_cc --test /dev/stdin <<< "(+ 1 2 3 4)" 2>&1; echo "exit=$?"
```
Expected: process does not crash. (This calls `+` with wrong arg count? Actually `+` takes varargs so this won't fail. Let me try something else.)

```bash
out/build/pscm-cmake/pscm_cc --test /dev/stdin <<< "(car 1 2)" 2>&1; echo "exit=$?"
```
Expected: error printed to stderr, process exits cleanly (exit 0).

- [ ] **Step 4: Commit**

```bash
git add src/c/eval.h
git commit -m "fix: replace exit(1) with eval_error in report_arg_mismatch"
```

---

### Task 6: Convert eval.cc exit(1) to eval_error

**Files:**
- Modify: `src/c/eval.cc:870-878`

- [ ] **Step 1: Replace exit(1) with eval_error in eval_with_env**

In `src/c/eval.cc`, inside `eval_with_env`, at line ~870-878, the error handler for unsupported expression types. Replace:

Old:
```cpp
    // Flush stderr to ensure all output is visible before exiting
    fflush(stderr);
    
    // Use exit instead of abort to ensure output is flushed
    exit(1);
  }
}
```

New:
```cpp
    // Flush stderr to ensure all output is visible
    fflush(stderr);
    
    eval_error("not supported expression type: %s", type_name);
  }
}
```

- [ ] **Step 2: Build**

```bash
ninja -C out/build/pscm-cmake pscm_cc
```

- [ ] **Step 3: Test**

```bash
out/build/pscm-cmake/pscm_cc --test /dev/stdin <<< "#f" 2>&1; echo "exit=$?"
```
Expected: `#f` evaluates fine (it's a valid self-evaluating form). Actually this case shouldn't be an error. Let me test the actual error path by passing something that would trigger it. The eval_with_env code only reaches that error handler for unsupported types. In practice, this shouldn't be reachable with valid Scheme input.

- [ ] **Step 4: Commit**

```bash
git add src/c/eval.cc
git commit -m "fix: replace exit(1) with eval_error in eval_with_env error path"
```

---

### Task 7: Convert quasiquote.cc exit(1) to eval_error

**Files:**
- Modify: `src/c/quasiquote.cc:10-18`

- [ ] **Step 1: Replace quasiquote_error with eval_error**

In `src/c/quasiquote.cc`, replace the `quasiquote_error` function entirely. Old:

```cpp
// Error handling helper for quasiquote
[[noreturn]] static void quasiquote_error(const char *format, ...) {
  va_list args;
  va_start(args, format);
  fprintf(stderr, "quasiquote error: ");
  vfprintf(stderr, format, args);
  fprintf(stderr, "\n");
  va_end(args);
  exit(1);
}
```

New:

```cpp
// Error handling helper for quasiquote — delegates to eval_error
// which routes through scm_throw (longjmp to nearest catch).
static void quasiquote_error(const char *format, ...) {
  va_list args;
  va_start(args, format);
  char msg[512];
  int pos = snprintf(msg, sizeof(msg), "quasiquote: ");
  vsnprintf(msg + pos, sizeof(msg) - pos, format, args);
  va_end(args);
  eval_error("%s", msg);
}
```

Remove `[[noreturn]]` (same reason as other sites — eval_error may return in no-catch fallback).

Add the include at the top of quasiquote.cc after existing includes:

```cpp
#include "error.h"
```

- [ ] **Step 2: Build**

```bash
ninja -C out/build/pscm-cmake pscm_cc
```

- [ ] **Step 3: Test**

```bash
out/build/pscm-cmake/pscm_cc --test /dev/stdin <<< "(quasiquote (unquote x y))" 2>&1; echo "exit=$?"
```
Expected: error message printed, process does not crash.

- [ ] **Step 4: Commit**

```bash
git add src/c/quasiquote.cc
git commit -m "fix: replace exit(1) with eval_error in quasiquote_error"
```

---

### Task 8: Convert print.cc exit(1) to eval_error

**Files:**
- Modify: `src/c/print.cc:285-301`

- [ ] **Step 1: Replace exit(1) with eval_error in _print_ast_with_context**

In `src/c/print.cc`, inside `_print_ast_with_context`, the unsupported-type error block around line 285-301. Replace:

Old:
```cpp
  fflush(stderr);
  exit(1);
}
```

New:
```cpp
  fflush(stderr);
  eval_error("unsupported type value %d in print", (int)ast->type);
}
```

Add `#include "error.h"` at the top of print.cc after existing includes.

- [ ] **Step 2: Build**

```bash
ninja -C out/build/pscm-cmake pscm_cc
```

- [ ] **Step 3: Commit**

```bash
git add src/c/print.cc
git commit -m "fix: replace exit(1) with eval_error in print error path"
```

---

### Task 9: Make eval_error fallback safe (no abort)

**Files:**
- Modify: `src/c/error.cc:231-245`

- [ ] **Step 1: Change eval_error abort() to return**

In `src/c/error.cc`, the `eval_error` function's else branch (when `g_error_key` is null) currently calls `abort()`. Replace:

Old:
```cpp
  if (g_error_key) {
    scm_throw(g_error_key, error_args_wrapped);
  } else {
    fprintf(stderr, "%s\n", full_message);
    fprintf(stderr, "\n=== Evaluation Call Stack ===\n");
    if (g_eval_stack) {
      print_eval_stack();
    } else {
      fprintf(stderr, "Call stack is empty (error occurred at top level)\n");
    }
    fprintf(stderr, "=== End of Call Stack ===\n");
    fflush(stderr);
    abort();
  }
}
```

New:
```cpp
  if (g_error_key) {
    scm_throw(g_error_key, error_args_wrapped);
  }
  // Fallthrough: if scm_throw did not longjmp (should not happen normally
  // since g_error_key is set during init and the API boundary wraps in
  // catch-all), print the error and return so the host stays alive.
  fprintf(stderr, "%s\n", full_message);
  fprintf(stderr, "\n=== Evaluation Call Stack ===\n");
  if (g_eval_stack) {
    print_eval_stack();
  } else {
    fprintf(stderr, "Call stack is empty (error occurred at top level)\n");
  }
  fprintf(stderr, "=== End of Call Stack ===\n");
  fflush(stderr);
}
```

Also remove `[[noreturn]]` from the `eval_error` declaration. In `src/c/error.h` (assuming it's declared there) and the definition in `error.cc`:

In `error.cc` line ~197, change:
```cpp
[[noreturn]] void eval_error(const char *format, ...) {
```
to:
```cpp
void eval_error(const char *format, ...) {
```

In `error.h` lines 27-28, remove `[[noreturn]]`:

```cpp
void eval_error(const char *format, ...);
void type_error(SCM *data, const char *expected_type);
```

- [ ] **Step 2: Build**

```bash
ninja -C out/build/pscm-cmake pscm_cc
```

- [ ] **Step 3: Commit**

```bash
git add src/c/error.cc src/c/error.h
git commit -m "fix: change eval_error fallback from abort to graceful return"
```

---

### Task 10: Add C API error test

**Files:**
- Create: `test/c_api/c_api_error.cpp`
- Modify: `test/c_api/CMakeLists.txt`

- [ ] **Step 1: Create test file**

Create `test/c_api/c_api_error.cpp`:

```cpp
#include "pscm_api.h"
#include <cassert>
#include <cstdio>
#include <cstring>

int main() {
  pscm_init();

  // Test 1: parse error returns NULL
  {
    SCM *result = pscm_eval_string("(define");
    assert(result == nullptr);
    const char *key = pscm_get_last_error_key();
    const char *msg = pscm_get_last_error_message();
    assert(key != nullptr);
    assert(strcmp(key, "error") == 0);
    assert(msg != nullptr);
    printf("PASS: parse error -> key='%s', msg='%s'\n", key, msg);
  }

  // Test 2: unbound variable returns NULL
  {
    SCM *result = pscm_eval_string("undefined-variable");
    assert(result == nullptr);
    const char *msg = pscm_get_last_error_message();
    assert(msg != nullptr);
    printf("PASS: unbound variable -> msg='%s'\n", msg);
  }

  // Test 3: wrong number of args returns NULL
  {
    SCM *result = pscm_eval_string("(car 1 2)");
    assert(result == nullptr);
    const char *msg = pscm_get_last_error_message();
    assert(msg != nullptr);
    printf("PASS: wrong arg count -> msg='%s'\n", msg);
  }

  // Test 4: valid code still returns non-NULL
  {
    SCM *result = pscm_eval_string("(+ 1 2)");
    assert(result != nullptr);
    assert((int64_t)result->value == 3);
    printf("PASS: valid code returns correct result\n");
  }

  // Test 5: quasiquote error returns NULL
  {
    SCM *result = pscm_eval_string("(quasiquote (unquote x y))");
    assert(result == nullptr);
    const char *msg = pscm_get_last_error_message();
    assert(msg != nullptr);
    printf("PASS: quasiquote error -> msg='%s'\n", msg);
  }

  // Test 6: pscm_parse error returns NULL
  {
    SCM *result = pscm_parse("(");
    assert(result == nullptr);
    const char *msg = pscm_get_last_error_message();
    assert(msg != nullptr);
    printf("PASS: pscm_parse error -> msg='%s'\n", msg);
  }

  printf("\nAll C API error tests passed!\n");
  return 0;
}
```

- [ ] **Step 2: Register test in CMakeLists.txt**

Add to `test/c_api/CMakeLists.txt` after the existing `c_api_module` lines:

```cmake
add_executable(c_api_error c_api_error.cpp ${pscm_sources})
target_include_directories(c_api_error PRIVATE "${PSCM_SOURCE_DIR}/src/c")
target_compile_options(c_api_error PRIVATE ${CMAKE_CXX_FLAGS})

add_test(NAME c_api_error COMMAND c_api_error)
```

- [ ] **Step 3: Build and run the test**

```bash
ninja -C out/build/pscm-cmake c_api_error && ./out/build/pscm-cmake/test/c_api/c_api_error
```
Expected: all 6 tests PASS

- [ ] **Step 4: Commit**

```bash
git add test/c_api/c_api_error.cpp test/c_api/CMakeLists.txt
git commit -m "test: add C API error handling integration tests"
```

---

### Note on throw.cc:312 (scm_uncaught_throw)

No separate task needed. After wrapping the public API in catch-all (`#t`), all
Scheme throws are caught — `scm_uncaught_throw` will only be reached if the
catch mechanism itself is broken, which is a logic bug. The existing `exit(1)`
is appropriate for that case.

### Task 11: Run all existing test suites, verify no regressions

- [ ] **Step 1: Full build**

```bash
ninja -C out/build/pscm-cmake
```
Expected: no build errors.

- [ ] **Step 2: Run all lit test suites**

```bash
ninja -C out/build/pscm-cmake check-base && \
ninja -C out/build/pscm-cmake check-cont && \
ninja -C out/build/pscm-cmake check-sicp && \
ninja -C out/build/pscm-cmake check-macro && \
ninja -C out/build/pscm-cmake check-core && \
ninja -C out/build/pscm-cmake check-gc
```
Expected: all 104 tests pass (100%).

- [ ] **Step 3: Run r4rs and r5rs tests**

```bash
out/build/pscm-cmake/pscm_cc --test test/r4rs/r4rstest.scm 2>&1 | tail -5 && \
out/build/pscm-cmake/pscm_cc --test test/r5rs/r5rstest.scm 2>&1 | tail -5
```
Expected: "Passed all tests" for both.

- [ ] **Step 4: Run C API tests**

```bash
ninja -C out/build/pscm-cmake && ctest --test-dir out/build/pscm-cmake -R c_api --output-on-failure
```
Expected: c_api_basic, c_api_module, c_api_error all pass.

- [ ] **Step 5: Commit (if any fixes needed)**

```bash
git add -u
git commit -m "chore: final verification and fixes after exit(1) elimination"
```
