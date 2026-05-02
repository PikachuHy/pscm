#include "pscm.h"

#include "eval.h"
#include "smob.h"
#include "throw.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

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


// scm_c_eval_string: Evaluate a C string containing Scheme code (compatible with Guile 1.8)
// Parses and evaluates all expressions in the string, returns the result of the last expression
SCM *scm_c_eval_string(const char *expr) {
  if (!expr) {
    return nullptr;
  }
  
  // Parse all expressions from the string
  SCM_List *exprs = parse_string(expr);
  if (!exprs) {
    return nullptr;
  }
  
  // Evaluate all expressions, return the last result
  if (!cont_base) {
    long stack_base;
    cont_base = &stack_base;
  }
  SCM *last_result = nullptr;
  SCM_List *current = exprs;
  while (current) {
    if (current->data) {
      last_result = eval_with_env(&g_env, current->data);
    }
    current = current->next;
  }
  
  // Return last result, or unspecified if no expressions
  return last_result ? last_result : scm_none();
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
  
  // Basic safety check: val should not be null
  if (!val) {
    eval_error("set!: evaluation of value returned null");
    return nullptr;
  }
  
  // First try to update in current environment (no parent search)
  // This ensures set! updates the binding in the current lexical scope
  auto entry = scm_env_search_entry(env, sym, /*search_parent=*/false);
  if (entry) {
    entry->value = val;
  } else {
    // Not found in current environment, search in parent environments
    // This handles cases where set! is used to update a binding in a parent scope
    entry = scm_env_search_entry(env, sym, /*search_parent=*/true);
    if (entry) {
      entry->value = val;
    } else {
      // Not found in environment, check if it's in the current module
      SCM *current_mod = scm_current_module();
      if (current_mod && is_module(current_mod)) {
        SCM_Module *module = cast<SCM_Module>(current_mod);
        SCM_Module *var_module = module_find_variable_module(module, sym);
        if (var_module) {
          // Variable exists in a module, update it
          scm_c_hash_set_eq(wrap(var_module->obarray), wrap(sym), val);
        } else {
          // Variable not found in any module, create new binding in environment
          scm_env_insert(env, sym, val, /*search_parent=*/false);
        }
      } else {
        // No module, create new binding in environment
        scm_env_insert(env, sym, val, /*search_parent=*/false);
      }
    }
  }
  
  if (debug_enabled) {
    SCM_DEBUG_EVAL("set! ");
    printf("%s to ", sym->data);
    print_ast(val);
    printf("\n");
  }
  // set! returns an unspecified value, which should not be printed
  return scm_none();
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

  // Push onto evaluation stack for call trace
  push_eval_stack(ast);

  // Save current context for error reporting
  SCM *old_context = g_current_eval_context;
  g_current_eval_context = ast;

// Helper macro to restore context before returning
#define RETURN_WITH_CONTEXT(val)                                                                                       \
  do {                                                                                                                 \
    g_current_eval_context = old_context;                                                                              \
    pop_eval_stack();                                                                                                  \
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
      // If macro expansion returns a non-pair value (e.g., a number), 
      // we need to handle it directly instead of trying to evaluate it as a list
      if (!is_pair(ast)) {
        RETURN_WITH_CONTEXT(ast);
      }
      // After macro expansion, we need to re-check the structure
      // If ast is a pair but l->data is not a symbol/proc/func, it might be
      // a self-evaluating value wrapped in a list, which should be evaluated directly
      // We need to recast l since ast might have changed
      l = cast<SCM_List>(ast);
      if (l && l->data && !is_sym(l->data) && !is_proc(l->data) && !is_func(l->data) && !is_cont(l->data) && !is_pair(l->data) && !l->next) {
        // This is a self-evaluating value (number, string, etc.) wrapped in a single-element list
        // Return it directly
        RETURN_WITH_CONTEXT(l->data);
      }
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
    else if (is_sym_val(l->data, "define-module")) {
      SCM *result = eval_define_module(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "use-modules")) {
      SCM *result = eval_use_modules(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "export")) {
      SCM *result = eval_export(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "re-export")) {
      SCM *result = eval_re_export(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "define-public")) {
      SCM *result = eval_define_public(env, l);
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
    else if (is_sym_val(l->data, "delay")) {
      SCM *result = eval_delay(env, l);
      RETURN_WITH_CONTEXT(result);
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
    else if (is_sym_val(l->data, "map-in-order")) {
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
      
      // Safety check: validate val before using it
      if (!val) {
        eval_error("symbol '%s' not found", sym->data);
        return nullptr;
      }
      
      // Check if val has a valid type before creating new list
      // Use try-catch to safely check type
      bool type_valid = false;
      try {
        if (val->type >= SCM::NONE && val->type <= SCM::MODULE) {
          type_valid = true;
        }
      } catch (...) {
        type_valid = false;
      }
      
      if (!type_valid) {
        fprintf(stderr, "Error: symbol '%s' resolved to invalid value with type %d (0x%x) at %p\n",
                sym->data, 
                (val ? (int)val->type : -1), 
                (val ? (unsigned int)val->type : 0), 
                (void *)val);
        fprintf(stderr, "  This suggests the symbol lookup returned a corrupted pointer.\n");
        fprintf(stderr, "  Valid type range: %d (NONE) to %d (MODULE)\n", 
                (int)SCM::NONE, (int)SCM::MODULE);
        fflush(stderr);
        eval_error("symbol '%s' resolved to corrupted value", sym->data);
        return nullptr;
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
  else if (is_smob(l->data)) {
    SCM_Smob *s = cast<SCM_Smob>(l->data);
    SCM_SmobDescriptor *desc = scm_get_smob_descriptor(s->tag);
    
    if (!desc) {
      eval_error("eval: invalid smob type");
      RETURN_WITH_CONTEXT(scm_none());
    }
    
    // Count arguments
    int arg_count = 0;
    SCM_List *arg_list = l->next;
    while (arg_list) {
      arg_count++;
      if (!arg_list->next) break;
      arg_list = arg_list->next;
    }
    
    // Call appropriate apply function
    SCM *result = scm_none();
    if (arg_count == 0 && desc->apply_0) {
      result = desc->apply_0(l->data);
    } else if (arg_count == 1 && desc->apply_1) {
      SCM *arg1 = eval_with_env(env, l->next->data);
      result = desc->apply_1(l->data, arg1);
    } else if (arg_count == 2 && desc->apply_2) {
      SCM *arg1 = eval_with_env(env, l->next->data);
      SCM *arg2 = eval_with_env(env, l->next->next->data);
      result = desc->apply_2(l->data, arg1, arg2);
    } else if (arg_count == 3 && desc->apply_3) {
      SCM *arg1 = eval_with_env(env, l->next->data);
      SCM *arg2 = eval_with_env(env, l->next->next->data);
      SCM *arg3 = eval_with_env(env, l->next->next->next->data);
      result = desc->apply_3(l->data, arg1, arg2, arg3);
    } else {
      eval_error("eval: smob does not support %d arguments", arg_count);
      RETURN_WITH_CONTEXT(scm_none());
    }
    
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
    if (func->name && strcmp(func->name->data, "map") == 0) {
      // For map, when called as a function value, arguments are already evaluated
      // We need to pass them to eval_map, which will handle them correctly
      // l->data is the map function, l->next contains the already-evaluated arguments
      SCM *result = eval_map(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    if (func->name && strcmp(func->name->data, "map-in-order") == 0) {
      // For map-in-order, when called as a function value, arguments are already evaluated
      // We need to pass them to eval_map, which will handle them correctly
      // l->data is the map-in-order function, l->next contains the already-evaluated arguments
      SCM *result = eval_map(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    if (func->name && (strcmp(func->name->data, "call-with-current-continuation") == 0 || 
                       strcmp(func->name->data, "call/cc") == 0)) {
      // For call-with-current-continuation, when called as a function value, arguments are already evaluated
      // We need to handle it specially, similar to how it's handled as a special form
      // l->data is the call-with-current-continuation function, l->next contains the already-evaluated arguments
      if (!l->next) {
        eval_error("call-with-current-continuation: requires 1 argument (procedure)");
        RETURN_WITH_CONTEXT(nullptr);
      }
      // The argument is already evaluated, so we can use it directly
      // But we need to create a continuation and call the procedure with it
      SCM *proc = l->next->data;
      if (!is_proc(proc) && !is_func(proc)) {
        eval_error("call-with-current-continuation: argument must be a procedure");
        RETURN_WITH_CONTEXT(nullptr);
      }
      // Create continuation and call procedure
      int first;
      auto cont = scm_make_continuation(&first);
      if (!first) {
        // Continuation was invoked, return the argument
        RETURN_WITH_CONTEXT(cont);
      }
      // Call procedure with continuation
      SCM_List *cont_list = make_list(cont);
      SCM_List proc_call;
      proc_call.data = proc;
      proc_call.next = cont_list;
      if (is_proc(proc)) {
        SCM *result = apply_procedure(env, cast<SCM_Procedure>(proc), cont_list);
        RETURN_WITH_CONTEXT(result);
      } else {
        // For function, evaluate arguments and call
        SCM_List *evaled_args = eval_list_with_env(env, cont_list);
        SCM_List func_call;
        func_call.data = proc;
        func_call.next = evaled_args;
        SCM *result = eval_with_func(cast<SCM_Function>(proc), &func_call);
        RETURN_WITH_CONTEXT(result);
      }
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
  else if (is_num(l->data) || is_float(l->data) || is_str(l->data) || is_char(l->data) || is_bool(l->data) || is_nil(l->data)) {
    // Self-evaluating values: numbers, strings, characters, booleans, nil
    // If this is a single-element list with a self-evaluating value, return it directly
    if (!l->next) {
      RETURN_WITH_CONTEXT(l->data);
    }
    // Otherwise, this is an error - can't have a list starting with a self-evaluating value
    eval_error("invalid expression: self-evaluating %s in non-tail position",
               get_type_name(l->data->type));
    return nullptr;
  }
  else {
    // Enhanced error reporting for unsupported expression types
    SCM *expr = l ? l->data : nullptr;
    const char *type_name = "unknown";
    const char *loc_str = nullptr;
    const char *list_loc_str = nullptr;
    
    // First, immediately print basic error info to stderr and flush
    // This ensures we see something even if accessing expr crashes
    fprintf(stderr, "<unknown location>: Error: not supported expression type: ");
    fflush(stderr);
    
    // Safely get type name - check if expr is valid before accessing
    if (expr) {
      // Check if expr is a valid pointer by checking if it's within reasonable bounds
      // This is a basic sanity check - in practice, if expr is corrupted, we might still crash
      // but at least we try to get the type safely
      try {
        type_name = get_type_name(expr->type);
        fprintf(stderr, "%s\n", type_name);
      } catch (...) {
        type_name = "corrupted";
        fprintf(stderr, "%s\n", type_name);
      }
    } else {
      type_name = get_type_name(SCM::NONE);
      fprintf(stderr, "%s\n", type_name);
    }
    fflush(stderr);
    
    // Now try to get location info - but if it crashes, we've at least printed the error
    // Add safety check: validate expr->type before accessing expr->source_loc
    if (expr) {
      // First check if type is valid before accessing source_loc
      // This prevents crashes from accessing corrupted memory
      bool type_valid = false;
      try {
        // Check if type is within valid enum range
        if (expr->type >= SCM::NONE && expr->type <= SCM::MODULE) {
          type_valid = true;
        }
      } catch (...) {
        // If accessing expr->type crashes, skip location info
        type_valid = false;
      }
      
      if (type_valid) {
        try {
          loc_str = get_source_location_str(expr);
        } catch (...) {
          // If accessing source_loc crashes, just skip location info
          loc_str = nullptr;
        }
      }
    }
    
    // Try to get source location from the list structure
    if (l) {
      try {
        SCM *list_wrapped = wrap(l);
        if (list_wrapped) {
          // Check if list_wrapped type is valid before accessing source_loc
          bool list_type_valid = false;
          try {
            if (list_wrapped->type >= SCM::NONE && list_wrapped->type <= SCM::MODULE) {
              list_type_valid = true;
            }
          } catch (...) {
            list_type_valid = false;
          }
          
          if (list_type_valid) {
            try {
              list_loc_str = get_source_location_str(list_wrapped);
            } catch (...) {
              list_loc_str = nullptr;
            }
          }
        }
      } catch (...) {
        // If wrap or accessing crashes, just skip
        list_loc_str = nullptr;
      }
    }
    
    // If still no location, try current eval context
    if (!loc_str && !list_loc_str && g_current_eval_context) {
      try {
        // Check if context type is valid before accessing source_loc
        bool ctx_type_valid = false;
        try {
          if (g_current_eval_context->type >= SCM::NONE && g_current_eval_context->type <= SCM::MODULE) {
            ctx_type_valid = true;
          }
        } catch (...) {
          ctx_type_valid = false;
        }
        
        if (ctx_type_valid) {
          try {
            loc_str = get_source_location_str(g_current_eval_context);
          } catch (...) {
            loc_str = nullptr;
          }
        }
      } catch (...) {
        // If accessing context crashes, just skip
        loc_str = nullptr;
      }
    }
    
    // If we got location info, print it (but we already printed error above)
    if (loc_str || list_loc_str) {
      fprintf(stderr, "  Location: %s\n", loc_str ? loc_str : list_loc_str);
    }
    
    // Print the problematic expression with its location
    // Use try-catch to handle potential crashes from corrupted pointers
    if (expr) {
      fprintf(stderr, "\n  Problematic expression");
      if (loc_str) {
        fprintf(stderr, " (at %s)", loc_str);
      }
      fprintf(stderr, ": ");
      try {
        print_ast_to_stderr(expr);
      } catch (...) {
        fprintf(stderr, "<unable to print - corrupted pointer>");
      }
    } else {
      fprintf(stderr, "\n  Expression is null");
    }
    
    // Print the full list structure for context with its location
    fprintf(stderr, "\n  Full expression");
    if (list_loc_str) {
      fprintf(stderr, " (at %s)", list_loc_str);
    }
    fprintf(stderr, ": ");
    try {
      if (l) {
        print_ast_to_stderr(wrap(l));
      } else {
        fprintf(stderr, "<list is null>");
      }
    } catch (...) {
      fprintf(stderr, "<unable to print - corrupted pointer>");
    }
    
    // Print the full expression being evaluated (if different from expr)
    if (g_current_eval_context) {
      try {
        SCM *ctx_wrapped = l ? wrap(l) : nullptr;
        if (g_current_eval_context != expr && g_current_eval_context != ctx_wrapped) {
          const char *ctx_loc = nullptr;
          try {
            ctx_loc = get_source_location_str(g_current_eval_context);
          } catch (...) {
            // Skip location if it crashes
          }
          fprintf(stderr, "\n  While evaluating");
          if (ctx_loc) {
            fprintf(stderr, " (at %s)", ctx_loc);
          }
          fprintf(stderr, ": ");
          try {
            print_ast_to_stderr(g_current_eval_context);
          } catch (...) {
            fprintf(stderr, "<unable to print - corrupted pointer>");
          }
        }
      } catch (...) {
        // If accessing context crashes, just skip it
      }
    }
    fprintf(stderr, "\n");
    
    // Print the evaluation call stack to show the evaluation path
    fprintf(stderr, "\n=== Evaluation Call Stack (showing how we got here) ===\n");
    if (g_eval_stack) {
      print_eval_stack();
    } else {
      fprintf(stderr, "Call stack is empty (error occurred at top level)\n");
    }
    fprintf(stderr, "=== End of Call Stack ===\n");
    
    // Print summary at the end for quick reference
    fprintf(stderr, "\n=== Error Summary ===\n");
    fprintf(stderr, "Error Type: not supported expression type: %s\n", type_name);
    if (loc_str) {
      fprintf(stderr, "Error Location: %s\n", loc_str);
    } else if (list_loc_str) {
      fprintf(stderr, "Error Location: %s\n", list_loc_str);
    } else {
      fprintf(stderr, "Error Location: <unknown> (check call stack above for details)\n");
    }
    fprintf(stderr, "Problematic Expression: ");
    if (expr) {
      try {
        print_ast_to_stderr(expr);
      } catch (...) {
        fprintf(stderr, "<unable to print - corrupted pointer>");
      }
    } else {
      fprintf(stderr, "<null>");
    }
    fprintf(stderr, "\n");
    
    // Flush stderr to ensure all output is visible
    fflush(stderr);

    eval_error("not supported expression type: %s", type_name);
    return nullptr;
  }
}

// Scheme eval function: (eval expr) -> evaluates expr in the current environment
SCM *scm_c_eval(SCM *expr) {
  // Evaluate the expression in the global environment
  return eval_with_env(&g_env, expr);
}

// defined?: Check if a symbol is defined in the current environment or module
// (defined? sym) -> #t if sym is defined, #f otherwise
SCM *scm_c_defined(SCM *sym) {
  if (!is_sym(sym)) {
    eval_error("defined?: expected symbol");
    return nullptr;  // Never reached, but satisfies compiler
  }
  
  SCM_Symbol *symbol = cast<SCM_Symbol>(sym);
  
  // 1. Check in lexical environment (search parent environments too)
  SCM_Environment::Entry *entry = scm_env_search_entry(&g_env, symbol, /*search_parent=*/true);
  if (entry) {
    return scm_bool_true();  // Found in environment
  }
  
  // 2. Check in current module
  SCM *current_mod = scm_current_module();
  if (current_mod && is_module(current_mod)) {
    SCM_Module *module = cast<SCM_Module>(current_mod);
    SCM *var = module_search_variable(module, symbol);
    if (var) {
      return scm_bool_true();  // Found in module (value can be #f, but it's still defined)
    }
  }
  
  return scm_bool_false();  // Not found
}

void init_eval() {
  scm_define_function("eval", 1, 0, 0, scm_c_eval);
  scm_define_function("defined?", 1, 0, 0, scm_c_defined);
}
