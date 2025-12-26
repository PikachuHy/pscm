#include "pscm.h"
#include "eval.h"

// Apply procedure with already-evaluated arguments (values)
// This is used by catch/throw handlers, where arguments are already evaluated
SCM *apply_procedure_with_values(SCM_Environment *env, SCM_Procedure *proc, SCM_List *args) {
  auto proc_env = make_env(proc->env);
  auto args_l = proc->args;
  // args are ALREADY evaluated, so we don't call eval_with_env on them
  
  // Check if the last parameter is a rest parameter (dotted pair: (a b . c))
  bool has_rest_param = false;
  SCM_Symbol *rest_param_sym = nullptr;
  SCM_List *penultimate_param = nullptr;
  bool only_rest_param = false;
  if (args_l) {
    SCM_List *last_param = args_l;
    SCM_List *penultimate = nullptr;
    while (last_param->next) {
      penultimate = last_param;
      last_param = last_param->next;
    }
    
    if (last_param->is_dotted && is_sym(last_param->data)) {
      has_rest_param = true;
      rest_param_sym = cast<SCM_Symbol>(last_param->data);
      penultimate_param = penultimate;
      only_rest_param = (penultimate == nullptr);
    }
  }
  
  // Bind regular parameters
  SCM_List *current_param = args_l;
  SCM_List *args_iter = args;
  
  // Special case: only rest parameter (e.g., (values . things))
  if (only_rest_param) {
    // Bind all arguments to the rest parameter (args are already evaluated)
    SCM *rest_val = args_iter ? wrap(args_iter) : scm_nil();
    scm_env_insert(proc_env, rest_param_sym, rest_val, /*search_parent=*/false);
    
    // Execute procedure body
    auto body = proc->body;
    return eval_with_list(proc_env, body);
  }
  
  // Bind regular parameters (not rest parameter)
  while (current_param && current_param != penultimate_param && !only_rest_param) {
    if (!is_sym(current_param->data)) {
      eval_error("procedure parameter must be a symbol");
    }
    SCM_Symbol *param_sym = cast<SCM_Symbol>(current_param->data);
    
    if (!args_iter) {
      eval_error("procedure called with too few arguments");
    }
    
    // args_iter->data is already evaluated
    scm_env_insert(proc_env, param_sym, args_iter->data, /*search_parent=*/false);
    
    current_param = current_param->next;
    args_iter = args_iter->next;
  }
  
  // Handle penultimate parameter before rest (e.g., for (key . args), bind key)
  if (has_rest_param && penultimate_param && !only_rest_param) {
    if (!is_sym(penultimate_param->data)) {
      eval_error("procedure parameter must be a symbol");
    }
    SCM_Symbol *param_sym = cast<SCM_Symbol>(penultimate_param->data);
    
    if (!args_iter) {
      eval_error("procedure called with too few arguments");
    }
    
    // args_iter->data is already evaluated
    scm_env_insert(proc_env, param_sym, args_iter->data, /*search_parent=*/false);
    args_iter = args_iter->next;
  }
  
  // Bind rest parameter if present
  if (has_rest_param) {
    // Collect remaining arguments into a list (they are already evaluated)
    SCM *rest_val = args_iter ? wrap(args_iter) : scm_nil();
    scm_env_insert(proc_env, rest_param_sym, rest_val, /*search_parent=*/false);
  } else {
    // No rest parameter - check for too many arguments
    if (args_iter) {
      eval_error("procedure called with too many arguments");
    }
  }
  
  // Execute procedure body
  auto body = proc->body;
  return eval_with_list(proc_env, body);
}

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
  // For (. rest), the structure is: (rest . nil) with is_dotted = true
  // The difference: in (a b . rest), we have 3 nodes, and the last node's data is the rest symbol
  // In (a b), we have 2 nodes, and the last node's data is b (not a rest symbol)
  // In (. rest), we have 1 node, and it has is_dotted = true
  bool has_rest_param = false;
  SCM_Symbol *rest_param_sym = nullptr;
  SCM_List *penultimate_param = nullptr;
  bool only_rest_param = false;  // True if the only parameter is a rest parameter
  if (args_l) {
    // Check if the last parameter is a rest parameter by checking is_dotted flag
    // For (a b . rest): the last node has is_dotted = true
    // For (a b c): all nodes have is_dotted = false
    // For (. rest): the only node has is_dotted = true
    SCM_List *last_param = args_l;
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
  SCM_List *current_param = args_l;
  
  // Special case: only rest parameter (e.g., (values . things))
  if (only_rest_param) {
    // Bind all arguments to the rest parameter
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
  } else {
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
  }
  
  // Check for argument mismatch (only if no rest parameter)
  if (!has_rest_param && (args_iter || current_param)) {
    // Construct the call expression for better error reporting
    SCM *call_expr = nullptr;
    if (proc->name) {
      // Create a list starting with the procedure name
      SCM_List *call_list = make_list(wrap(proc->name));
      SCM_List *call_tail = call_list;
      SCM_List *arg_iter = original_args;
      while (arg_iter) {
        SCM_List *arg_node = make_list(arg_iter->data);
        call_tail->next = arg_node;
        call_tail = arg_node;
        arg_iter = arg_iter->next;
      }
      call_expr = wrap(call_list);
    }
    report_arg_mismatch(proc->args, original_args, "Procedure", call_expr, proc->name);
  }
  
  return eval_with_list(proc_env, proc->body);
}

SCM *scm_c_procedure_name(SCM *arg) {
  if (!is_proc(arg)) {
    eval_error("procedure-name: expected procedure");
    return nullptr;
  }
  SCM_Procedure *proc = cast<SCM_Procedure>(arg);
  if (proc->name) {
    return wrap(proc->name);
  }
  return scm_bool_false();
}

void init_procedure() {
  scm_define_function("procedure-name", 1, 0, 0, scm_c_procedure_name);
}

