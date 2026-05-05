# Parser: TeXmacs Guile 1.8 Syntax Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add support for three Guile 1.8 syntax extensions used by TeXmacs files: backslash-escaped characters in symbols, `#{...}#` reader macro, and defensive handling of bare `(unquote)`.

**Architecture:** Three independent fixes in the parser/evaluator: (1) symbol tokenizer gets `\` escape support, (2) expression parser gets `#{` handler, (3) quasiquote expander handles bare unquote gracefully.

**Tech Stack:** pscm (C++20 Scheme interpreter), recursive-descent parser in parse.cc

---

### Task 1: Add backslash escape handling in symbol parser

**Files:**
- Modify: `src/c/parse.cc:808-814`

- [ ] **Step 1: Replace the symbol character loop**

Current code at lines 808-814:

```cpp
  while (*p->pos && 
         (isalnum((unsigned char)*p->pos) || 
          strchr("!$%&*+-./:<=>?@^_~|", *p->pos) != nullptr)) {
    len++;
    p->pos++;
    p->column++;
  }
```

Replace with:

```cpp
  while (*p->pos && 
         (isalnum((unsigned char)*p->pos) || 
          *p->pos == '\\' ||
          strchr("!$%&*+-./:<=>?@^_~|", *p->pos) != nullptr)) {
    if (*p->pos == '\\') {
      // Guile 1.8 backslash escape: next character is part of the symbol literally.
      // Example: left\{  -> symbol name "left{"
      len++; p->pos++; p->column++;  // consume backslash
      if (*p->pos && !isspace((unsigned char)*p->pos) && 
          *p->pos != '(' && *p->pos != ')' && *p->pos != ';') {
        len++; p->pos++; p->column++;  // consume escaped character
      }
    } else {
      len++;
      p->pos++;
      p->column++;
    }
  }
```

- [ ] **Step 2: Build and run existing tests (no regressions)**

```bash
ninja -C out/build/pscm-cmake
ctest --test-dir out/build/pscm-cmake --output-on-failure
```

Expected: all 45 tests pass.

- [ ] **Step 3: Quick manual test of backslash escape**

```bash
echo "(display 'left\\{)" | out/build/pscm-cmake/pscm_cc
```

Expected: prints `left{` (not a parse error).

- [ ] **Step 4: Commit**

```bash
git add src/c/parse.cc
git commit -m "feat: add backslash escape in symbol names (Guile 1.8 compat)"
```

---

### Task 2: Add `#{...}#` reader macro for arbitrary symbol names

**Files:**
- Modify: `src/c/parse.cc` — add handler between `#:keyword` and char literal (after line 1211)

- [ ] **Step 1: Add `#{...}#` handler in parse_expr()**

After the `#:keyword` check block (lines 1202-1211), add:

```cpp

  // #{...}# symbol reader syntax (Guile 1.8)
  // Creates a symbol whose name is the literal content between #{ and }#
  // Example: #{Apple Symbols}# -> symbol "Apple Symbols"
  if (p->pos[0] == '#' && p->pos[1] == '{') {
    int sym_start_line = p->line;
    int sym_start_col = p->column;
    p->pos += 2;  // skip #{
    p->column += 2;
    const char *content_start = p->pos;
    int brace_depth = 1;
    while (*p->pos && brace_depth > 0) {
      if (*p->pos == '{') brace_depth++;
      else if (*p->pos == '}') brace_depth--;
      if (brace_depth > 0) {
        if (*p->pos == '\n') { p->line++; p->column = 0; }
        else p->column++;
        p->pos++;
      }
    }
    if (brace_depth != 0) {
      parse_error(p, "unterminated #{...}# reader syntax");
      return nullptr;
    }
    int content_len = (p->pos) - content_start;
    p->pos++;  // skip closing }
    p->column++;
    if (p->pos[0] != '#') {
      parse_error(p, "expected # after } in #{...}#");
      return nullptr;
    }
    p->pos++;  // skip closing #
    p->column++;
    
    char *name = new char[content_len + 1];
    memcpy(name, content_start, content_len);
    name[content_len] = '\0';
    SCM *sym = create_sym(name, content_len);
    set_source_location(sym, p->filename, sym_start_line, sym_start_col);
    delete[] name;
    return sym;
  }
```

- [ ] **Step 2: Build and test**

```bash
ninja -C out/build/pscm-cmake
ctest --test-dir out/build/pscm-cmake --output-on-failure
```

Expected: all 45 tests pass.

- [ ] **Step 3: Quick manual test**

```bash
echo "(display '#{hello world}#)" | out/build/pscm-cmake/pscm_cc
echo "(display '#{}#)" | out/build/pscm-cmake/pscm_cc
```

Expected: prints `hello world` and `#{}#` (empty symbol).

- [ ] **Step 4: Commit**

```bash
git add src/c/parse.cc
git commit -m "feat: add #{...}# reader macro for arbitrary symbol names"
```

---

### Task 3: Defensive handling of bare (unquote) in quasiquote

**Files:**
- Modify: `src/c/quasiquote.cc:197-200`

- [ ] **Step 1: Replace error with graceful handling**

Current code at lines 197-200:

```cpp
  if (is_unquote_form(p)) {
    SCM *arg = get_form_arg(p);
    if (!arg) {
      quasiquote_error("unquote requires an argument");
```

Replace with:

```cpp
  if (is_unquote_form(p)) {
    SCM *arg = get_form_arg(p);
    if (!arg) {
      // Bare (unquote) with no argument — can happen during TeXmacs
      // macro expansion when , is used as pattern-matching syntax.
      // Return unspecified value instead of erroring.
      return scm_none();
```

- [ ] **Step 2: Build and run all tests**

```bash
ninja -C out/build/pscm-cmake
ctest --test-dir out/build/pscm-cmake --output-on-failure
ninja -C out/build/pscm-cmake check-base check-cont check-macro check-core
```

Expected: all tests pass, no regressions in macro/quasiquote tests.

- [ ] **Step 3: Run key TeXmacs tests**

```bash
ctest --test-dir out/build/pscm-cmake -R 'load_kernel|load_utils|cross_module' --output-on-failure
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/c/quasiquote.cc
git commit -m "fix: treat bare (unquote) as noop in quasiquote expansion"
```

---

## Summary

| Task | What | File | Independent? |
|------|------|------|-------------|
| 1 | `\` escape in symbols | parse.cc:808-814 | Yes |
| 2 | `#{...}#` reader macro | parse.cc:1211+ | Yes |
| 3 | Bare `(unquote)` noop | quasiquote.cc:200 | Yes |

All three tasks are independent and can be dispatched in parallel.
