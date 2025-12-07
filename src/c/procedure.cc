#include "pscm.h"
#include "eval.h"

// Helper function to apply procedure with arguments
SCM *apply_procedure(SCM_Environment *env, SCM_Procedure *proc, SCM_List *args) {
  auto proc_env = make_env(proc->env);
  auto args_l = proc->args;
  // Save original args for error reporting - create a copy of the list structure
  // (we only need the structure, not the values, for error reporting)
  SCM_List *original_args = args; // Save original args pointer for error reporting
  SCM_List *args_iter = args;     // Use separate iterator for the loop
  
  // Check if the last parameter is a rest parameter (dotted pair: (a b . c))
  // For (a b), the structure is: (a . (b . nil))
  // For (a b . rest), the structure is: (a . (b . (rest . nil)))
  // The difference: in (a b . rest), we have 3 nodes, and the last node's data is the rest symbol
  // In (a b), we have 2 nodes, and the last node's data is b (not a rest symbol)
  bool has_rest_param = false;
  SCM_Symbol *rest_param_sym = nullptr;
  SCM_List *penultimate_param = nullptr;
  if (args_l) {
    // Count parameters and find the last one
    int param_count = 0;
    SCM_List *last_param = args_l;
    while (last_param->next) {
      last_param = last_param->next;
      param_count++;
    }
    param_count++; // Include the first parameter
    
    // A rest parameter requires at least 2 parameters before it: (a b . rest)
    // And the structure must have the rest symbol as the last node's data
    // For (a b . rest): param_count >= 3, last_param->data is the rest symbol
    // For (a b): param_count == 2, last_param->data is b (regular param)
    if (param_count >= 3 && is_sym(last_param->data)) {
      // Check if this is really a rest parameter by verifying the structure
      // Find the penultimate parameter
      SCM_List *penultimate = args_l;
      while (penultimate->next != last_param) {
        penultimate = penultimate->next;
      }
      // If penultimate is also a symbol (regular param), then last is rest param
      if (is_sym(penultimate->data)) {
        has_rest_param = true;
        rest_param_sym = cast<SCM_Symbol>(last_param->data);
        penultimate_param = penultimate;
      }
    }
  }
  
  // Bind regular parameters
  SCM_List *current_param = args_l;
  while (args_iter && current_param) {
    // Check if this is the penultimate parameter before rest parameter
    if (has_rest_param && current_param == penultimate_param) {
      // Bind the penultimate parameter
      assert(is_sym(current_param->data));
      auto arg_sym = cast<SCM_Symbol>(current_param->data);
      auto arg_val = eval_with_env(env, args_iter->data);
      scm_env_insert(proc_env, arg_sym, arg_val, /*search_parent=*/false);
      if (debug_enabled) {
        SCM_DEBUG_EVAL("bind func arg ");
        printf("%s to ", arg_sym->data);
        print_ast(arg_val);
        printf("\n");
      }
      args_iter = args_iter->next;
      current_param = current_param->next;
      
      // Now bind all remaining arguments to the rest parameter
      SCM_List rest_args_dummy = make_list_dummy();
      SCM_List *rest_args_tail = &rest_args_dummy;
      while (args_iter) {
        SCM *arg_val = eval_with_env(env, args_iter->data);
        SCM_List *node = make_list(arg_val);
        rest_args_tail->next = node;
        rest_args_tail = node;
        args_iter = args_iter->next;
      }
      SCM *rest_args = rest_args_dummy.next ? wrap(rest_args_dummy.next) : scm_nil();
      scm_env_insert(proc_env, rest_param_sym, rest_args, /*search_parent=*/false);
      if (debug_enabled) {
        SCM_DEBUG_EVAL("bind rest arg ");
        printf("%s to ", rest_param_sym->data);
        print_ast(rest_args);
        printf("\n");
      }
      break;
    }
    
    assert(is_sym(current_param->data));
    auto arg_sym = cast<SCM_Symbol>(current_param->data);
    auto arg_val = eval_with_env(env, args_iter->data);
    scm_env_insert(proc_env, arg_sym, arg_val, /*search_parent=*/false);
    if (debug_enabled) {
      SCM_DEBUG_EVAL("bind func arg ");
      printf("%s to ", arg_sym->data);
      print_ast(arg_val);
      printf("\n");
    }
    args_iter = args_iter->next;
    current_param = current_param->next;
  }
  
  // Check for argument mismatch (only if no rest parameter)
  if (!has_rest_param && (args_iter || current_param)) {
    report_arg_mismatch(proc->args, original_args);
  }
  
  return eval_with_list(proc_env, proc->body);
}

