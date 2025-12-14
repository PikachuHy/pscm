#include "pscm.h"
#include "eval.h"

// Helper function to expand a macro call
SCM *expand_macro_call(SCM_Environment *env, SCM_Macro *macro, SCM_List *args, SCM *original_call) {
  // Create macro environment (use macro's definition environment)
  SCM_Environment *macro_env = make_env(macro->env);

  // Bind macro arguments (pass syntax objects directly, without evaluation)
  SCM_List *macro_args = macro->transformer->args;
  SCM_List *actual_args = args;
  
  // Check if the last parameter is a rest parameter (dotted pair: (a b . c) or (. rest))
  // Similar to apply_procedure, we need to handle rest parameters specially
  bool has_rest_param = false;
  SCM_Symbol *rest_param_sym = nullptr;
  SCM_List *penultimate_param = nullptr;
  bool only_rest_param = false;  // True if the only parameter is a rest parameter
  if (macro_args) {
    // Check if the last parameter is a rest parameter by checking is_dotted flag
    SCM_List *last_param = macro_args;
    SCM_List *penultimate = nullptr;
    while (last_param->next) {
      penultimate = last_param;
      last_param = last_param->next;
    }
    
    // Check if last parameter is marked as dotted (rest parameter)
    if (last_param->is_dotted && is_sym(last_param->data)) {
      has_rest_param = true;
      rest_param_sym = cast<SCM_Symbol>(last_param->data);
      penultimate_param = penultimate;
      only_rest_param = (penultimate == nullptr);  // Only rest param if no penultimate
    }
  }
  
  // Bind regular parameters
  SCM_List *current_param = macro_args;
  
  // Special case: only rest parameter (e.g., (define-macro (test . args) ...))
  if (only_rest_param) {
    // Bind all arguments to the rest parameter (as syntax objects, not evaluated)
    SCM_List rest_args_dummy = make_list_dummy();
    SCM_List *rest_args_tail = &rest_args_dummy;
    while (actual_args) {
      // Pass the argument as-is (syntax object), not evaluated
      SCM_List *node = make_list(actual_args->data);
      rest_args_tail->next = node;
      rest_args_tail = node;
      actual_args = actual_args->next;
    }
    SCM *rest_args = rest_args_dummy.next ? wrap(rest_args_dummy.next) : scm_nil();
    scm_env_insert(macro_env, rest_param_sym, rest_args, /*search_parent=*/false);
    if (debug_enabled) {
      SCM_DEBUG_EVAL("bind macro rest arg ");
      printf("%s to ", rest_param_sym->data);
      print_ast(rest_args);
      printf("\n");
    }
  } else {
    // Bind regular parameters and handle rest parameter if present
    while (actual_args && current_param) {
      // Check if this is the penultimate parameter before rest parameter
      if (has_rest_param && current_param == penultimate_param) {
        // Bind the penultimate parameter
        assert(is_sym(current_param->data));
        auto param_sym = cast<SCM_Symbol>(current_param->data);
        // Pass the argument as-is (syntax object), not evaluated
        scm_env_insert(macro_env, param_sym, actual_args->data, /*search_parent=*/false);
        if (debug_enabled) {
          SCM_DEBUG_EVAL("bind macro arg ");
          printf("%s to ", param_sym->data);
          print_ast(actual_args->data);
          printf("\n");
        }
        actual_args = actual_args->next;
        current_param = current_param->next;
        
        // Now bind all remaining arguments to the rest parameter
        SCM_List rest_args_dummy = make_list_dummy();
        SCM_List *rest_args_tail = &rest_args_dummy;
        while (actual_args) {
          // Pass the argument as-is (syntax object), not evaluated
          SCM_List *node = make_list(actual_args->data);
          rest_args_tail->next = node;
          rest_args_tail = node;
          actual_args = actual_args->next;
        }
        SCM *rest_args = rest_args_dummy.next ? wrap(rest_args_dummy.next) : scm_nil();
        scm_env_insert(macro_env, rest_param_sym, rest_args, /*search_parent=*/false);
        if (debug_enabled) {
          SCM_DEBUG_EVAL("bind macro rest arg ");
          printf("%s to ", rest_param_sym->data);
          print_ast(rest_args);
          printf("\n");
        }
        break;
      }
      
      assert(is_sym(current_param->data));
      auto param_sym = cast<SCM_Symbol>(current_param->data);
      // Pass the argument as-is (syntax object), not evaluated
      scm_env_insert(macro_env, param_sym, actual_args->data, /*search_parent=*/false);
      if (debug_enabled) {
        SCM_DEBUG_EVAL("bind macro arg ");
        printf("%s to ", param_sym->data);
        print_ast(actual_args->data);
        printf("\n");
      }
      actual_args = actual_args->next;
      current_param = current_param->next;
    }
  }
  
  // Check for argument mismatch (only if no rest parameter)
  if (!has_rest_param && (actual_args || current_param)) {
    report_arg_mismatch(macro->transformer->args, args, "Macro", original_call, macro->name);
  }

  // Call the macro transformer (evaluate in macro environment)
  SCM *expanded = eval_with_list(macro_env, macro->transformer->body);

  // Copy source location from original macro call to expanded result
  // This ensures errors in macro-expanded code point to the original macro call location
  if (original_call && expanded) {
    // First, try to copy recursively to preserve structure
    copy_source_location_recursive(expanded, original_call);
    // Also ensure the top-level node has source location
    if (!expanded->source_loc && original_call->source_loc) {
      copy_source_location(expanded, original_call);
    }
  }

  if (debug_enabled) {
    SCM_DEBUG_EVAL("macro expanded to ");
    print_ast(expanded);
    printf("\n");
  }

  return expanded;
}

// Helper function to expand macros in an expression
SCM *expand_macros(SCM_Environment *env, SCM *ast) {
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
    SCM *val = scm_env_exist(env, sym);

    if (val && is_macro(val)) {
      // Found a macro, expand it
      SCM_Macro *macro = cast<SCM_Macro>(val);
      SCM *expanded = expand_macro_call(env, macro, l->next, ast);
      // Recursively expand the result
      return expand_macros(env, expanded);
    }
  }

  // Recursively expand each element in the list
  SCM_List dummy = make_list_dummy();
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
    scm->source_loc = nullptr;  // Initialize to nullptr
    // Copy source location from original list
    copy_source_location(scm, ast);
    return scm;
  }
  SCM *result = scm_nil();
  copy_source_location(result, ast);
  return result;
}

// Helper function for define-macro special form
SCM *eval_define_macro(SCM_Environment *env, SCM_List *l) {
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

  // Check if proc_sig->next is a single symbol (rest parameter like (test . args))
  // In this case, we need to convert it to a dotted pair structure
  SCM_List *args_list = proc_sig->next;
  if (args_list && is_sym(args_list->data) && !args_list->next) {
    // This is a rest parameter: (test . args) -> args_list is just (args)
    // We need to mark it as a dotted pair
    args_list->is_dotted = true;
  }

  SCM_Procedure *transformer = make_proc(name, args_list, l->next->next, env);

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

