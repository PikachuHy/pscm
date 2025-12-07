#include "pscm.h"
#include "eval.h"

// Helper function for apply special form
// (apply proc arg1 arg2 ... args-list) -> apply proc to (arg1 arg2 ... args-list elements)
SCM *eval_apply(SCM_Environment *env, SCM_List *l) {
  if (!l->next) {
    eval_error("apply: requires at least 2 arguments (procedure and arguments list)");
  }
  
  // Evaluate the procedure
  SCM *proc = eval_with_env(env, l->next->data);
  
  if (!is_proc(proc) && !is_func(proc)) {
    eval_error("apply: first argument must be a procedure");
  }
  
  // Collect all arguments before the last one
  SCM_List *args_before_last = l->next->next;
  SCM_List *last_arg_node = nullptr;
  
  // Find the last argument (which should be a list)
  if (!args_before_last) {
    eval_error("apply: requires at least 2 arguments (procedure and arguments list)");
  }
  
  SCM_List *current = args_before_last;
  while (current->next) {
    current = current->next;
  }
  last_arg_node = current;
  
  // Evaluate the last argument (should be a list)
  SCM *last_arg = eval_with_env(env, last_arg_node->data);
  if (!is_pair(last_arg) && !is_nil(last_arg)) {
    eval_error("apply: last argument must be a list");
  }
  
  // Build the combined argument list
  // First, evaluate all arguments before the last one
  SCM_List args_dummy = make_list_dummy();
  SCM_List *args_tail = &args_dummy;
  
  current = args_before_last;
  while (current != last_arg_node) {
    SCM *arg_val = eval_with_env(env, current->data);
    SCM_List *node = make_list(arg_val);
    args_tail->next = node;
    args_tail = node;
    current = current->next;
  }
  
  // Then append all elements from the last argument list
  if (is_pair(last_arg)) {
    SCM_List *last_list = cast<SCM_List>(last_arg);
    while (last_list) {
      // Wrap each element in quote to prevent double evaluation
      SCM *quoted_elem = scm_list2(scm_sym_quote(), last_list->data);
      SCM_List *node = make_list(quoted_elem);
      args_tail->next = node;
      args_tail = node;
      last_list = last_list->next;
    }
  }
  
  // Apply the procedure with the combined arguments
  if (is_proc(proc)) {
    SCM_Procedure *proc_obj = cast<SCM_Procedure>(proc);
    return apply_procedure(env, proc_obj, args_dummy.next);
  } else if (is_func(proc)) {
    SCM_Function *func_obj = cast<SCM_Function>(proc);
    SCM_List *evaled_args = eval_list_with_env(env, args_dummy.next);
    SCM_List func_call;
    func_call.data = proc;
    func_call.next = evaled_args;
    return eval_with_func(func_obj, &func_call);
  } else {
    eval_error("apply: first argument must be a procedure");
    return nullptr; // Never reached
  }
}

