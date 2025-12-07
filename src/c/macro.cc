#include "pscm.h"
#include "eval.h"

// Helper function to expand a macro call
SCM *expand_macro_call(SCM_Environment *env, SCM_Macro *macro, SCM_List *args, SCM *original_call) {
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
    SCM *val = scm_env_search(env, sym);

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

