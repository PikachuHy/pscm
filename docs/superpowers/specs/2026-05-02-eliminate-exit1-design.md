# Eliminate exit(1) from Recoverable Error Paths

## Goal

All recoverable Scheme errors return NULL from the public C API (`pscm_eval`,
`pscm_eval_string`, `pscm_parse`) instead of killing the host process. The host
can then query the last error via dedicated API functions.

## Non-goals

- Thread safety (this project is single-threaded)
- Signal safety for SIGSEGV/SIGABRT paths
- Fixing GC resource exhaustion behavior

## Architecture

Wrap every public API entry point in `scm_c_catch` with catch-all tag (`#t`).
The handler stores error details in globals and returns NULL. The existing
`scm_c_catch` / `scm_throw` / setjmp/longjmp infrastructure carries the errors;
the problem is just that some error sites bypass it and the API boundary lacks
a safety net.

```
Host calls pscm_eval("(+ 1 x)")   ;; x is unbound
  -> scm_c_catch(#t, eval_body, error_handler)
    -> eval_with_env executes, encounters unbound x
      -> eval_error("unbound variable: x")
        -> scm_throw('error, "unbound variable: x")
          -> longjmp to outermost catch frame
    -> error_handler: store tag + message, return nullptr
  -> returns nullptr to host
  -> host calls pscm_get_last_error_message() -> "unbound variable: x"
```

## Changes per exit(1) Site

| File:line | Current | Change |
|-----------|---------|--------|
| `parse.cc:256` | `exit(1)` on syntax error | `eval_error("Parse error: %s", msg)` |
| `eval.h:156` | `exit(1)` on wrong arg count | `eval_error("Wrong number of arguments: ...")` |
| `eval.cc:876` | `exit(1)` in `eval_with_env` unsupported-expr error path | `eval_error("not supported expression type: %s", type_name)` |
| `quasiquote.cc:18` | `exit(1)` on quasiquote error | `eval_error("quasiquote: %s", msg)` |
| `throw.cc:312` | `exit(1)` on uncaught throw | Store error info; keep abort as last resort |
| `print.cc:299` | `exit(1)` on corrupted AST | `eval_error` (not signal-unsafe) |
| `abort.cc:70` | `exit(1)` in SIGABRT handler | **Keep** — signal handlers cannot safely call throw/longjmp |

## Public API Changes

### Modified functions

- `pscm_eval(SCM *ast)` — wraps body in `scm_c_catch(#t, ...)`, returns NULL on error
- `pscm_eval_string(const char *code)` — wraps in catch, returns NULL on error
- `pscm_parse(const char *code)` — wraps in catch, returns NULL on parse error

### New functions

```c
// Returns the error key symbol name (e.g. "error", "misc-error"), or NULL if no error.
const char *pscm_get_last_error_key(void);

// Returns the error message string, or NULL if no error.
const char *pscm_get_last_error_message(void);
```

Error state is stored in global variables (single-threaded runtime).

## Error Handling Inside scm_c_catch

The handler registered at the API boundary:

1. Copies the error tag and message string (the message is a GC-managed string
   that may be freed after the catch returns)
2. Stores copies in global `g_last_error_*` variables
3. Returns `nullptr` — the API function propagates this NULL to the host

## eval_error Fallback

When `g_error_key` is not set (should not happen after init, but defense in depth),
`eval_error` currently calls `abort()`. Change this to:

1. Print the error message and call stack to stderr (existing behavior)
2. Return without aborting (the caller must handle the error return)

This way, even if the catch mechanism somehow fails, the process stays alive.

## What Does NOT Change

- `abort.cc:70` / `print_stacktrace` — SIGABRT/SIGSEGV/SIGILL handlers.
  These run in signal context. No safe way to call longjmp or allocate memory.
  Exit remains the only safe option.
- GC resource exhaustion (`gc.cc` `abort()` calls) — mmap failure or segment
  pool exhaustion means the runtime is in an unrecoverable state. abort remains.
- `exit.cc` `scm_c_exit` — this is the Scheme `(exit)` function. Keeps exit(3)
  as that is its documented behavior.

## Testing Strategy

1. **Unit tests**: For each converted exit(1) site, add a test that triggers the
   error and verifies the host gets NULL (not a crash).
2. **C API test**: `test/c_api/c_api_error.cpp` — calls `pscm_eval_string` with
   invalid code, asserts NULL return, checks `pscm_get_last_error_message()`.
3. **Existing test suites** must continue to pass at 100%.
