#include "pscm.h"

#include "eval.h"
#include <stdarg.h>

SCM *eval_with_list(SCM_Environment *env, SCM_List *l) {
  assert(l);
  SCM *ret = nullptr;
  while (l) {
    ret = eval_with_env(env, l->data);
    l = l->next;
  }
  return ret;
}

SCM_List *eval_list_with_env(SCM_Environment *env, SCM_List *l) {
  SCM_List dummy = make_list_dummy();
  SCM_List *it = &dummy;
  while (l) {
    auto val = eval_with_env(env, l->data);
    auto next = make_list(val);
    it->next = next;
    it = it->next;
    l = l->next;
  }
  return dummy.next;
}


// Error handling helper with context
static SCM *g_current_eval_context = nullptr;

[[noreturn]] void eval_error(const char *format, ...) {
  va_list args;
  va_start(args, format);

  // Print source location if available
  bool printed_location = false;
  if (g_current_eval_context) {
    const char *loc_str = get_source_location_str(g_current_eval_context);
    if (loc_str) {
      fprintf(stderr, "%s: ", loc_str);
      printed_location = true;
    }
    fprintf(stderr, "Error while evaluating: ");
    print_ast(g_current_eval_context);
    fprintf(stderr, "\n");
    if (!printed_location) {
      fprintf(stderr, "  (no source location available)\n");
    }
    fprintf(stderr, "  ");
  }
  else {
    fprintf(stderr, "%s:%d: ", __FILE__, __LINE__);
  }

  vfprintf(stderr, format, args);
  fprintf(stderr, "\n");
  va_end(args);
  exit(1);
}

// Helper functions for special forms
static SCM *eval_quote(SCM_List *l) {
  return l->next ? l->next->data : scm_nil();
}

SCM *eval_quasiquote(SCM_Environment *env, SCM_List *l);

static SCM *eval_set(SCM_Environment *env, SCM_List *l) {
  assert(l->next && is_sym(l->next->data));
  auto sym = cast<SCM_Symbol>(l->next->data);
  auto val = eval_with_env(env, l->next->next->data);
  scm_env_insert(env, sym, val);
  if (debug_enabled) {
    SCM_DEBUG_EVAL("set! ");
    printf("%s to ", sym->data);
    print_ast(val);
    printf("\n");
  }
  return scm_nil();
}

static SCM *eval_lambda(SCM_Environment *env, SCM_List *l) {
  auto proc_sig = cast<SCM_List>(l->next->data);
  auto proc = make_proc(nullptr, proc_sig, l->next->next, env);
  auto ret = wrap(proc);
  if (debug_enabled) {
    SCM_DEBUG_EVAL("create proc ");
    print_ast(ret);
    printf(" from ");
    print_list(l);
    printf("\n");
  }
  return ret;
}

static SCM *eval_define(SCM_Environment *env, SCM_List *l) {
  if (l->next && is_sym(l->next->data)) {
    // Define variable: (define var value)
    SCM_Symbol *varname = cast<SCM_Symbol>(l->next->data);
    SCM_DEBUG_EVAL("define variable %s\n", varname->data);
    auto val = eval_with_env(env, l->next->next->data);
    assert(val);
    if (is_proc(val)) {
      auto proc = cast<SCM_Procedure>(val);
      if (proc->name == nullptr) {
        proc->name = varname;
        SCM_DEBUG_EVAL("define proc from lambda\n");
      }
    }
    scm_env_insert(env, varname, val);
    return scm_none();
  }
  // Define procedure: (define (name args...) body...)
  SCM_List *proc_sig = cast<SCM_List>(l->next->data);
  SCM_DEBUG_EVAL("define a procedure");
  if (!is_sym(proc_sig->data)) {
    eval_error("not supported define form");
  }
  SCM_Symbol *proc_name = cast<SCM_Symbol>(proc_sig->data);
  SCM_DEBUG_EVAL(" %s with params ", proc_name->data);
  if (debug_enabled) {
    printf("(");
    if (proc_sig->next) {
      print_ast(proc_sig->next->data);
    }
    printf(")\n");
  }
  auto proc = make_proc(proc_name, proc_sig->next, l->next->next, env);
  SCM *ret = wrap(proc);
  scm_env_insert(env, proc_name, ret);
  return ret;
}


// Helper function to count list length
static int count_list_length(SCM_List *l) {
  int count = 0;
  while (l) {
    count++;
    l = l->next;
  }
  return count;
}

// Helper function to lookup symbol in environment
static SCM *lookup_symbol(SCM_Environment *env, SCM_Symbol *sym) {
  auto val = scm_env_search(env, sym);
  if (!val) {
    eval_error("symbol '%s' not found", sym->data);
  }
  return val;
}

// Helper function for if special form
static SCM *eval_if(SCM_Environment *env, SCM_List *l, SCM **ast) {
  assert(l->next);
  auto pred = eval_with_env(env, l->next->data);
  if (is_true(pred)) {
    *ast = l->next->next->data;
    return nullptr; // Signal to continue evaluation
  }
  if (l->next->next->next) {
    *ast = l->next->next->next->data;
    return nullptr; // Signal to continue evaluation
  }
  return scm_none();
}

// Helper function for call/cc special form
static SCM *eval_call_cc(SCM_Environment *env, SCM_List *l, SCM **ast) {
  assert(l->next);
  auto proc = eval_with_env(env, l->next->data);
  int first;
  auto cont = scm_make_continuation(&first);
  SCM_DEBUG_CONT("jump back: ");
  if (!first) {
    if (debug_enabled) {
      printf("cont is ");
      print_ast(cont);
      printf("\n");
    }
    return cont;
  }
  *ast = scm_list2(proc, cont);
  if (debug_enabled) {
    print_ast(*ast);
    printf("\n");
  }
  return nullptr; // Signal to continue evaluation
}

// Helper function for cond special form
static SCM *eval_cond(SCM_Environment *env, SCM_List *l, SCM **ast) {
  assert(l->next);
  for (auto it = l->next; it; it = it->next) {
    auto clause = cast<SCM_List>(it->data);
    if (debug_enabled) {
      SCM_DEBUG_EVAL("eval cond clause ");
      print_list(clause);
      printf("\n");
    }
    if (is_sym_val(clause->data, "else")) {
      return eval_with_list(env, clause->next);
    }
    auto pred = eval_with_env(env, clause->data);
    if (is_bool(pred) && is_false(pred)) {
      continue;
    }
    if (!clause->next) {
      return scm_bool_true();
    }
    if (is_sym_val(clause->next->data, "=>")) {
      assert(clause->next->next);
      *ast = scm_list2(clause->next->next->data, scm_list2(scm_sym_quote(), pred));
      return nullptr; // Signal to continue evaluation
    }
    return eval_with_list(env, clause->next);
  }
  return scm_none();
}

// Helper function for for-each special form
static SCM *eval_for_each(SCM_Environment *env, SCM_List *l) {
  assert(l->next);
  auto f = eval_with_env(env, l->next->data);
  auto proc = cast<SCM_Procedure>(f);
  int arg_count = count_list_length(proc->args);
  SCM_List dummy = make_list_dummy();
  auto result_tail = &dummy;
  l = l->next->next;

  SCM_List args_dummy = make_list_dummy();
  auto args_iter = &args_dummy;

  // Evaluate all argument lists
  for (int i = 0; i < arg_count; i++) {
    if (!l) {
      eval_error("args count not match, require %d, but got %d", arg_count, i);
    }
    auto item = make_list(eval_with_env(env, l->data));
    result_tail->next = item;
    result_tail = item;
    l = l->next;
    auto arg = make_list();
    args_iter->next = arg;
    args_iter = arg;
  }
  args_dummy.data = f;
  assert(arg_count == 1);
  auto arg_l = cast<SCM_List>(dummy.next->data);
  while (arg_l) {
    args_dummy.next->data = arg_l->data;
    SCM call_expr;
    call_expr.type = SCM::LIST;
    call_expr.value = &args_dummy;
    if (debug_enabled) {
      SCM_DEBUG_EVAL("for-each ");
      print_ast(f);
      printf(" ");
      print_ast(arg_l->data);
      printf("\n");
    }
    eval_with_env(env, &call_expr);
    arg_l = arg_l->next;
  }
  return scm_none();
}

// Helper function to apply procedure with arguments
SCM *apply_procedure(SCM_Environment *env, SCM_Procedure *proc, SCM_List *args) {
  auto proc_env = make_env(proc->env);
  auto args_l = proc->args;
  // Save original args for error reporting - create a copy of the list structure
  // (we only need the structure, not the values, for error reporting)
  SCM_List *original_args = args; // Save original args pointer for error reporting
  SCM_List *args_iter = args;     // Use separate iterator for the loop
  while (args_iter && args_l) {
    assert(is_sym(args_l->data));
    auto arg_sym = cast<SCM_Symbol>(args_l->data);
    auto arg_val = eval_with_env(env, args_iter->data);
    scm_env_insert(proc_env, arg_sym, arg_val, /*search_parent=*/false);
    if (debug_enabled) {
      SCM_DEBUG_EVAL("bind func arg ");
      printf("%s to ", arg_sym->data);
      print_ast(arg_val);
      printf("\n");
    }
    args_iter = args_iter->next;
    args_l = args_l->next;
  }
  if (args_iter || args_l) {
    report_arg_mismatch(proc->args, original_args);
  }
  return eval_with_list(proc_env, proc->body);
}

SCM *eval_with_env(SCM_Environment *env, SCM *ast) {
  assert(env);
  assert(ast);

  // Save current context for error reporting
  SCM *old_context = g_current_eval_context;
  g_current_eval_context = ast;

// Helper macro to restore context before returning
#define RETURN_WITH_CONTEXT(val)                                                                                       \
  do {                                                                                                                 \
    g_current_eval_context = old_context;                                                                              \
    return (val);                                                                                                      \
  } while (0)

entry:
  SCM_DEBUG_EVAL("eval ");
  if (debug_enabled) {
    print_ast(ast);
    printf("\n");
  }
  if (!is_pair(ast)) {
    if (is_sym(ast)) {
      SCM_Symbol *sym = cast<SCM_Symbol>(ast);
      SCM *result = lookup_symbol(env, sym);
      RETURN_WITH_CONTEXT(result);
    }
    RETURN_WITH_CONTEXT(ast);
  }
  SCM_List *l = cast<SCM_List>(ast);
  assert(l->data);
  if (is_sym(l->data)) {
    SCM_Symbol *sym = cast<SCM_Symbol>(l->data);

    // Check if this is a macro call (before checking special forms)
    SCM *val = scm_env_exist(env, sym);
    if (val && is_macro(val)) {
      // Found a macro, expand it and continue evaluation
      SCM_Macro *macro = cast<SCM_Macro>(val);
      SCM *expanded = expand_macro_call(env, macro, l->next, ast);
      // Recursively expand the result, then evaluate
      ast = expand_macros(env, expanded);
      goto entry;
    }

    if (is_sym_val(l->data, "define")) {
      SCM *result = eval_define(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "define-macro")) {
      SCM *result = eval_define_macro(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "let")) {
      ast = expand_let(ast);
      goto entry;
    }
    else if (is_sym_val(l->data, "let*")) {
      ast = expand_letstar(ast);
      goto entry;
    }
    else if (is_sym_val(l->data, "letrec")) {
      ast = expand_letrec(ast);
      goto entry;
    }
    else if (is_sym_val(l->data, "call/cc") || is_sym_val(l->data, "call-with-current-continuation")) {
      auto ret = eval_call_cc(env, l, &ast);
      if (ret) {
        RETURN_WITH_CONTEXT(ret);
      }
      goto entry;
    }
    else if (is_sym_val(l->data, "lambda")) {
      SCM *result = eval_lambda(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "set!")) {
      SCM *result = eval_set(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "quote")) {
      SCM *result = eval_quote(l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "quasiquote")) {
      SCM *result = eval_quasiquote(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "if")) {
      auto ret = eval_if(env, l, &ast);
      if (ret) {
        RETURN_WITH_CONTEXT(ret);
      }
      goto entry;
    }
    else if (is_sym_val(l->data, "cond")) {
      auto ret = eval_cond(env, l, &ast);
      if (ret) {
        RETURN_WITH_CONTEXT(ret);
      }
      goto entry;
    }
    else if (is_sym_val(l->data, "for-each")) {
      SCM *result = eval_for_each(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "do")) {
      SCM *result = eval_do(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "map")) {
      SCM *result = eval_map(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else {
      // Variable reference: resolve symbol and build call expression
      // (val was already looked up above for macro check)
      if (!val) {
        val = lookup_symbol(env, sym);
      }
      auto new_list = make_list(val);
      new_list->next = l->next;
      ast = wrap(new_list);
      goto entry;
    }
  }
  else if (is_cont(l->data)) {
    SCM *cont_arg = scm_nil();
    if (l->next) {
      cont_arg = eval_with_env(env, l->next->data);
    }
    scm_dynthrow(l->data, cont_arg);
  }
  else if (is_proc(l->data)) {
    auto proc = cast<SCM_Procedure>(l->data);
    SCM *result = apply_procedure(env, proc, l->next);
    RETURN_WITH_CONTEXT(result);
  }
  else if (is_func(l->data)) {
    auto func = cast<SCM_Function>(l->data);
    auto func_argl = eval_list_with_env(env, l->next);
    if (debug_enabled) {
      SCM_DEBUG_EVAL(" ");
      printf("before eval args: ");
      print_list(l->next);
      printf("\n");
      printf("after eval args: ");
      print_list(func_argl);
      printf("\n");
    }
    l->next = func_argl;
    SCM *result = eval_with_func(func, l);
    RETURN_WITH_CONTEXT(result);
  }
  else if (is_pair(l->data)) {
    // Nested list: evaluate first element and continue
    auto f = eval_with_env(env, l->data);
    auto new_l = make_list(f);
    new_l->next = l->next;
    ast = wrap(new_l);
    goto entry;
  }
  else {
    eval_error("not supported expression type");
  }
}
