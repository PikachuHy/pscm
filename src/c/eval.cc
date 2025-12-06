#include "pscm.h"
#include <stdarg.h>

SCM_List _make_list_dummy() {
  SCM_List dummy;
  dummy.data = nullptr;
  dummy.next = nullptr;
  return dummy;
}

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
  SCM_List dummy = _make_list_dummy();
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

SCM *eval_with_func_1(SCM_Function *func, SCM *arg1) {
  typedef SCM *(*func_1)(SCM *);
  auto f = (func_1)func->func_ptr;
  return f(arg1);
}

SCM *eval_with_func_2(SCM_Function *func, SCM *arg1, SCM *arg2) {
  typedef SCM *(*func_2)(SCM *, SCM *);
  auto f = (func_2)func->func_ptr;
  return f(arg1, arg2);
}

// Error handling helper
[[noreturn]] static void eval_error(const char *format, ...) {
  va_list args;
  va_start(args, format);
  fprintf(stderr, "%s:%d ", __FILE__, __LINE__);
  vfprintf(stderr, format, args);
  fprintf(stderr, "\n");
  va_end(args);
  exit(1);
}

// Helper functions for special forms
static SCM *eval_quote(SCM_List *l) {
  return l->next ? l->next->data : scm_nil();
}

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

SCM *eval_with_func(SCM_Function *func, SCM_List *l) {
  if (debug_enabled) {
    SCM_DEBUG_EVAL("eval func ");
    printf("%s with ", func->name->data);
    print_list(l->next);
    printf("\n");
  }
  if (func->n_args == 1) {
    assert(l->next);
    return eval_with_func_1(func, l->next->data);
  }
  if (func->n_args == 2) {
    assert(l->next && l->next->next);
    return eval_with_func_2(func, l->next->data, l->next->next->data);
  }
  if (func->n_args == -1 && func->generic) {
    return reduce(
        [func](SCM *lhs, SCM *rhs) {
          return eval_with_func_2(func, lhs, rhs);
        },
        func->generic, l->next);
  }
  if (func->n_args == -2) {
    // Variable argument function (like list)
    typedef SCM *(*func_var)(SCM_List *);
    auto f = (func_var)func->func_ptr;
    return f(l->next);
  }
  eval_error("not supported function: %s", func->name->data);
  return nullptr; // Never reached, but satisfies compiler
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
  SCM_List dummy = _make_list_dummy();
  auto result_tail = &dummy;
  l = l->next->next;

  SCM_List args_dummy = _make_list_dummy();
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

// Helper function to update do loop variables
static void update_do_variables(SCM_Environment *do_env, SCM_List *var_update_list) {
  auto it = var_update_list;
  while (it->next) {
    it = it->next;
    auto var_update_expr = cast<SCM_List>(it->data);
    auto var_name = cast<SCM_Symbol>(var_update_expr->data);
    auto var_update_step = var_update_expr->next->data;

    auto new_var_val = eval_with_env(do_env, var_update_step);
    if (debug_enabled) {
      SCM_DEBUG_EVAL("eval do step ... ");
      print_ast(var_update_step);
      printf(" --> ");
      print_ast(new_var_val);
      printf("\n");
    }
    scm_env_insert(do_env, var_name, new_var_val);
  }
}

// Helper function for do special form
static SCM *eval_do(SCM_Environment *env, SCM_List *l) {
  assert(l->next && l->next->next && l->next->next->next);
  auto var_init_l = cast<SCM_List>(l->next->data);
  auto test_clause = l->next->next->data;
  auto body_clause = l->next->next->next;

  if (debug_enabled) {
    SCM_DEBUG_EVAL("eval do\n");
    printf("var: ");
    print_list(var_init_l);
    printf("\n");
    printf("test: ");
    print_ast(test_clause);
    printf("\n");
    printf("cmd: ");
    print_list(body_clause);
    printf("\n");
  }
  auto do_env = make_env(env);

  auto var_init_it = var_init_l;
  SCM_List var_update_dummy = _make_list_dummy();
  auto var_update_it = &var_update_dummy;

  while (var_init_it) {
    auto var_init_expr = cast<SCM_List>(var_init_it->data);
    auto var_name = cast<SCM_Symbol>(var_init_expr->data);
    auto var_init_val = eval_with_env(env, var_init_expr->next->data);
    auto var_update_step = var_init_expr->next->next->data;

    scm_env_insert(do_env, var_name, var_init_val);
    var_update_it->next = make_list(scm_list2(wrap(var_name), var_update_step));
    var_update_it = var_update_it->next;
    var_update_it->next = nullptr;
    var_init_it = var_init_it->next;
  }

  auto ret = eval_with_env(do_env, car(test_clause));
  while (is_false(ret)) {
    eval_list_with_env(do_env, body_clause);
    update_do_variables(do_env, &var_update_dummy);
    ret = eval_with_env(do_env, car(test_clause));
  }
  return scm_none();
}

// Helper function to report argument mismatch error
[[noreturn]] static void report_arg_mismatch(SCM_List *expected, SCM_List *got) {
  fprintf(stderr, "args not match\n");
  fprintf(stderr, "expect ");
  print_list(expected);
  fprintf(stderr, "\n");
  fprintf(stderr, "but got ");
  print_list(got);
  fprintf(stderr, "\n");
  exit(1);
}

// Helper function to apply procedure with arguments
static SCM *apply_procedure(SCM_Environment *env, SCM_Procedure *proc, SCM_List *args) {
  auto proc_env = make_env(proc->env);
  auto args_l = proc->args;
  while (args && args_l) {
    assert(is_sym(args_l->data));
    auto arg_sym = cast<SCM_Symbol>(args_l->data);
    auto arg_val = eval_with_env(env, args->data);
    scm_env_insert(proc_env, arg_sym, arg_val, /*search_parent=*/false);
    if (debug_enabled) {
      SCM_DEBUG_EVAL("bind func arg ");
      printf("%s to ", arg_sym->data);
      print_ast(arg_val);
      printf("\n");
    }
    args = args->next;
    args_l = args_l->next;
  }
  if (args || args_l) {
    report_arg_mismatch(proc->args, args);
  }
  return eval_with_list(proc_env, proc->body);
}

// Helper function to expand a macro call
static SCM *expand_macro_call(SCM_Environment *env, SCM_Macro *macro, SCM_List *args) {
  // Create macro environment (use macro's definition environment)
  SCM_Environment *macro_env = make_env(macro->env);

  // Bind macro arguments (pass syntax objects directly, without evaluation)
  SCM_List *macro_args = macro->transformer->args;
  SCM_List *actual_args = args;

  while (macro_args && actual_args) {
    SCM_Symbol *param = cast<SCM_Symbol>(macro_args->data);
    // Pass the argument as-is (syntax object), not evaluated
    scm_env_insert(macro_env, param, actual_args->data, /*search_parent=*/false);
    if (debug_enabled) {
      SCM_DEBUG_EVAL("bind macro arg ");
      printf("%s to ", param->data);
      print_ast(actual_args->data);
      printf("\n");
    }
    macro_args = macro_args->next;
    actual_args = actual_args->next;
  }

  if (macro_args || actual_args) {
    report_arg_mismatch(macro->transformer->args, args);
  }

  // Call the macro transformer (evaluate in macro environment)
  SCM *expanded = eval_with_list(macro_env, macro->transformer->body);

  if (debug_enabled) {
    SCM_DEBUG_EVAL("macro expanded to ");
    print_ast(expanded);
    printf("\n");
  }

  return expanded;
}

// Helper function to expand macros in an expression
static SCM *expand_macros(SCM_Environment *env, SCM *ast) {
  if (!is_pair(ast)) {
    return ast; // Atoms don't need expansion
  }

  SCM_List *l = cast<SCM_List>(ast);
  if (!l->data) {
    return ast;
  }

  // Check if this is a macro call
  if (is_sym(l->data)) {
    SCM_Symbol *sym = cast<SCM_Symbol>(l->data);
    SCM *val = scm_env_search(env, sym);

    if (val && is_macro(val)) {
      // Found a macro, expand it
      SCM_Macro *macro = cast<SCM_Macro>(val);
      SCM *expanded = expand_macro_call(env, macro, l->next);
      // Recursively expand the result
      return expand_macros(env, expanded);
    }
  }

  // Recursively expand each element in the list
  SCM_List dummy = _make_list_dummy();
  SCM_List *tail = &dummy;
  SCM_List *current = l;

  while (current) {
    SCM *expanded = expand_macros(env, current->data);
    SCM_List *node = make_list(expanded);
    tail->next = node;
    tail = node;
    current = current->next;
  }

  if (dummy.next) {
    SCM *scm = new SCM();
    scm->type = SCM::LIST;
    scm->value = dummy.next;
    return scm;
  }
  return scm_nil();
}

// Helper function for define-macro special form
static SCM *eval_define_macro(SCM_Environment *env, SCM_List *l) {
  if (!l->next) {
    eval_error("define-macro: missing arguments");
  }

  if (is_sym(l->next->data)) {
    // (define-macro name transformer)
    SCM_Symbol *name = cast<SCM_Symbol>(l->next->data);
    if (!l->next->next) {
      eval_error("define-macro: missing transformer");
    }
    SCM *transformer = eval_with_env(env, l->next->next->data);

    if (!is_proc(transformer)) {
      eval_error("define-macro: transformer must be a procedure");
    }

    SCM_Macro *macro = new SCM_Macro();
    macro->name = name;
    macro->transformer = cast<SCM_Procedure>(transformer);
    macro->env = env;

    SCM *macro_scm = wrap(macro);
    scm_env_insert(env, name, macro_scm);

    if (debug_enabled) {
      SCM_DEBUG_EVAL("define-macro ");
      printf("%s\n", name->data);
    }

    return scm_none();
  }

  // (define-macro (name args...) body...)
  if (!is_pair(l->next->data)) {
    eval_error("define-macro: invalid macro definition");
  }
  SCM_List *proc_sig = cast<SCM_List>(l->next->data);
  if (!is_sym(proc_sig->data)) {
    eval_error("define-macro: macro name must be a symbol");
  }
  SCM_Symbol *name = cast<SCM_Symbol>(proc_sig->data);

  SCM_Procedure *transformer = make_proc(name, proc_sig->next, l->next->next, env);

  SCM_Macro *macro = new SCM_Macro();
  macro->name = name;
  macro->transformer = transformer;
  macro->env = env;

  SCM *macro_scm = wrap(macro);
  scm_env_insert(env, name, macro_scm);

  if (debug_enabled) {
    SCM_DEBUG_EVAL("define-macro ");
    printf("%s\n", name->data);
  }

  return scm_none();
}

SCM *eval_with_env(SCM_Environment *env, SCM *ast) {
  assert(env);
  assert(ast);
entry:
  SCM_DEBUG_EVAL("eval ");
  if (debug_enabled) {
    print_ast(ast);
    printf("\n");
  }
  if (!is_pair(ast)) {
    if (is_sym(ast)) {
      SCM_Symbol *sym = cast<SCM_Symbol>(ast);
      return lookup_symbol(env, sym);
    }
    return ast;
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
      SCM *expanded = expand_macro_call(env, macro, l->next);
      // Recursively expand the result, then evaluate
      ast = expand_macros(env, expanded);
      goto entry;
    }

    if (is_sym_val(l->data, "define")) {
      return eval_define(env, l);
    }
    else if (is_sym_val(l->data, "define-macro")) {
      return eval_define_macro(env, l);
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
      if (ret)
        return ret;
      goto entry;
    }
    else if (is_sym_val(l->data, "lambda")) {
      return eval_lambda(env, l);
    }
    else if (is_sym_val(l->data, "set!")) {
      return eval_set(env, l);
    }
    else if (is_sym_val(l->data, "quote")) {
      return eval_quote(l);
    }
    else if (is_sym_val(l->data, "if")) {
      auto ret = eval_if(env, l, &ast);
      if (ret)
        return ret;
      goto entry;
    }
    else if (is_sym_val(l->data, "cond")) {
      auto ret = eval_cond(env, l, &ast);
      if (ret)
        return ret;
      goto entry;
    }
    else if (is_sym_val(l->data, "for-each")) {
      return eval_for_each(env, l);
    }
    else if (is_sym_val(l->data, "do")) {
      return eval_do(env, l);
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
    return apply_procedure(env, proc, l->next);
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
    return eval_with_func(func, l);
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
