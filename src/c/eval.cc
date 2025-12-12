#include "pscm.h"

#include "eval.h"
#include <stdarg.h>

// Helper function to print AST to stderr
static void print_ast_to_stderr(SCM *ast) {
  // Temporarily redirect stdout to stderr for print_ast
  // This is safe in single-threaded context
  fflush(stdout);
  FILE *saved_stdout = stdout;
  stdout = stderr;
  print_ast(ast);
  fflush(stderr);
  stdout = saved_stdout;
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

// Helper function to get type name as string
static const char *get_type_name(SCM::Type type) {
  switch (type) {
    case SCM::NONE: return "none";
    case SCM::NIL: return "nil";
    case SCM::LIST: return "pair/list";
    case SCM::PROC: return "procedure";
    case SCM::CONT: return "continuation";
    case SCM::FUNC: return "function";
    case SCM::NUM: return "number";
    case SCM::FLOAT: return "float";
    case SCM::CHAR: return "character";
    case SCM::BOOL: return "boolean";
    case SCM::SYM: return "symbol";
    case SCM::STR: return "string";
    case SCM::MACRO: return "macro";
    case SCM::HASH_TABLE: return "hash-table";
    case SCM::RATIO: return "ratio";
    case SCM::VECTOR: return "vector";
    default: return "unknown";
  }
}

[[noreturn]] void type_error(SCM *data, const char *expected_type) {
  // Try to get source location from the data itself first
  const char *loc_str = nullptr;
  if (data) {
    loc_str = get_source_location_str(data);
  }
  
  // If not available, try current eval context
  if (!loc_str && g_current_eval_context) {
    loc_str = get_source_location_str(g_current_eval_context);
  }
  
  if (loc_str) {
    fprintf(stderr, "%s: ", loc_str);
  } else {
    fprintf(stderr, "<unknown location>: ");
  }
  
  fprintf(stderr, "Type error: expected %s, but got ", expected_type);
  
  if (data) {
    const char *actual_type = get_type_name(data->type);
    fprintf(stderr, "%s", actual_type);
    fprintf(stderr, "\n  Value: ");
    print_ast(data);
    fprintf(stderr, "\n");
  } else {
    fprintf(stderr, "null\n");
  }
  
  // Also print current eval context if available and different
  if (g_current_eval_context && g_current_eval_context != data) {
    const char *ctx_loc = get_source_location_str(g_current_eval_context);
    if (ctx_loc) {
      fprintf(stderr, "  While evaluating at %s: ", ctx_loc);
    } else {
      fprintf(stderr, "  While evaluating: ");
    }
    print_ast(g_current_eval_context);
    fprintf(stderr, "\n");
  }
  
  exit(1);
}

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

SCM *eval_lambda(SCM_Environment *env, SCM_List *l) {
  SCM *param_spec = l->next->data;
  SCM_List *proc_sig = nullptr;
  
  // Check if parameter is a single symbol (e.g., (lambda x body))
  // In Scheme, (lambda x body) is equivalent to (lambda (. x) body)
  if (is_sym(param_spec)) {
    // Convert single symbol to a rest parameter list: (. symbol)
    auto rest_sym = cast<SCM_Symbol>(param_spec);
    SCM_List *rest_param_node = make_list(param_spec);
    rest_param_node->is_dotted = true;  // Mark as rest parameter
    proc_sig = rest_param_node;
  } else if (is_pair(param_spec) || is_nil(param_spec)) {
    // Normal parameter list (list or nil)
    proc_sig = cast<SCM_List>(param_spec);
  } else {
    type_error(param_spec, "symbol, pair, or nil");
  }
  
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


// Helper function to count list length
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
  // In Scheme, only #f is falsy, everything else is truthy
  if (is_truthy(pred)) {
    *ast = l->next->next->data;
    return nullptr; // Signal to continue evaluation
  }
  if (l->next->next->next) {
    *ast = l->next->next->next->data;
    return nullptr; // Signal to continue evaluation
  }
  return scm_none();
}

// Helper function for and special form
static SCM *eval_and(SCM_Environment *env, SCM_List *l, SCM **ast) {
  if (!l->next) {
    // No arguments, return #t
    return scm_bool_true();
  }
  SCM_List *current = l->next;
  // Evaluate all expressions except the last one
  while (current && current->next) {
    SCM *result = eval_with_env(env, current->data);
    // If any expression evaluates to #f (falsy), return #f immediately
    // In Scheme, only #f is falsy, everything else is truthy
    if (is_falsy(result)) {
      return scm_bool_false();
    }
    current = current->next;
  }
  // Evaluate and return the last expression
  if (current) {
    *ast = current->data;
    return nullptr; // Signal to continue evaluation
  }
  return scm_bool_true();
}

// Helper function for or special form
static SCM *eval_or(SCM_Environment *env, SCM_List *l, SCM **ast) {
  if (!l->next) {
    // No arguments, return #f
    return scm_bool_false();
  }
  SCM_List *current = l->next;
  // Evaluate expressions until we find one that's truthy
  while (current && current->next) {
    SCM *result = eval_with_env(env, current->data);
    // If any expression evaluates to truthy (not #f), return it immediately
    // In Scheme, only #f is falsy, everything else is truthy
    if (is_truthy(result)) {
      return result;
    }
    current = current->next;
  }
  // Evaluate and return the last expression
  if (current) {
    *ast = current->data;
    return nullptr; // Signal to continue evaluation
  }
  return scm_bool_false();
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
      // Keywords (symbols starting with ':') are self-evaluating
      if (sym->data && sym->data[0] == ':') {
        RETURN_WITH_CONTEXT(ast);
      }
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
    else if (is_sym_val(l->data, "and")) {
      auto ret = eval_and(env, l, &ast);
      if (ret) {
        RETURN_WITH_CONTEXT(ret);
      }
      goto entry;
    }
    else if (is_sym_val(l->data, "or")) {
      auto ret = eval_or(env, l, &ast);
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
    else if (is_sym_val(l->data, "case")) {
      SCM *result = eval_case(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "begin")) {
      // begin: evaluate all expressions in sequence, return the last one
      if (!l->next) {
        // No expressions, return #f
        RETURN_WITH_CONTEXT(scm_bool_false());
      }
      SCM_List *current = l->next;
      SCM *result = scm_bool_false();
      // Evaluate all expressions except the last one
      while (current && current->next) {
        eval_with_env(env, current->data);
        current = current->next;
      }
      // Evaluate and return the last expression
      if (current) {
        result = eval_with_env(env, current->data);
      }
      RETURN_WITH_CONTEXT(result);
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
    else if (is_sym_val(l->data, "apply")) {
      SCM *result = eval_apply(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "call-with-values")) {
      SCM *result = eval_call_with_values(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "dynamic-wind")) {
      SCM *result = eval_dynamic_wind(env, l);
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
    return nullptr;  // Never reached, but satisfies compiler
  }
  else if (is_proc(l->data)) {
    auto proc = cast<SCM_Procedure>(l->data);
    SCM *result = apply_procedure(env, proc, l->next);
    RETURN_WITH_CONTEXT(result);
  }
  else if (is_func(l->data)) {
    auto func = cast<SCM_Function>(l->data);
    // Special handling for apply: it needs access to environment and handles argument evaluation itself
    if (func->name && strcmp(func->name->data, "apply") == 0) {
      // For apply, when called as a function value, arguments are already evaluated
      // We need to pass them to eval_apply, which will handle them correctly
      // l->data is the apply function, l->next contains the already-evaluated arguments
      SCM *result = eval_apply(env, l);
      RETURN_WITH_CONTEXT(result);
    }
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
    // Enhanced error reporting for unsupported expression types
    SCM *expr = l->data;
    const char *type_name = get_type_name(expr ? expr->type : SCM::NONE);
    const char *loc_str = nullptr;
    
    // Try to get source location from the expression or current context
    if (expr) {
      loc_str = get_source_location_str(expr);
    }
    if (!loc_str && g_current_eval_context) {
      loc_str = get_source_location_str(g_current_eval_context);
    }
    
    // Print location if available
    if (loc_str) {
      fprintf(stderr, "%s: ", loc_str);
    } else {
      fprintf(stderr, "<unknown location>: ");
    }
    
    // Print error message with type information
    fprintf(stderr, "Error: not supported expression type: %s", type_name);
    
    // Print the problematic expression
    if (expr) {
      fprintf(stderr, "\n  Expression value: ");
      print_ast_to_stderr(expr);
    } else {
      fprintf(stderr, "\n  Expression is null");
    }
    
    // Print the full expression being evaluated (if different from expr)
    if (g_current_eval_context) {
      SCM *ctx_wrapped = wrap(l);
      if (g_current_eval_context != expr && g_current_eval_context != ctx_wrapped) {
        fprintf(stderr, "\n  While evaluating: ");
        print_ast_to_stderr(g_current_eval_context);
      }
    }
    
    // Print the full list structure for context
    fprintf(stderr, "\n  Full expression: ");
    print_ast_to_stderr(wrap(l));
    fprintf(stderr, "\n");
    
    abort();
    return nullptr;  // Never reached, but satisfies compiler
  }
}

// Scheme eval function: (eval expr) -> evaluates expr in the current environment
extern SCM_Environment g_env;

SCM *scm_c_eval(SCM *expr) {
  // Evaluate the expression in the global environment
  return eval_with_env(&g_env, expr);
}

void init_eval() {
  scm_define_function("eval", 1, 0, 0, scm_c_eval);
}
