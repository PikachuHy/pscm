#include "pscm.h"
#include "eval.h"

// Wrapper function for apply as a builtin function
// This function is called when apply is used as a value (not as a special form)
// It receives already-evaluated arguments, but apply needs to handle evaluation itself
// So we create a dummy function that will be handled specially in eval.cc
SCM *scm_c_apply(SCM_List *args) {
  // This function should never be called directly
  // It's registered as a placeholder, and eval.cc will handle it specially
  eval_error("apply: internal error - should be handled as special form");
  return nullptr;
}

// Helper function for apply special form
// (apply proc arg1 arg2 ... args-list) -> apply proc to (arg1 arg2 ... args-list elements)
// When called as a function value, l->next contains already-evaluated arguments
SCM *eval_apply(SCM_Environment *env, SCM_List *l) {
  if (!l->next) {
    eval_error("apply: requires at least 2 arguments (procedure and arguments list)");
  }
  
  // Arguments are NOT evaluated when coming from eval.cc (whether as symbol or function value)
  // They are only evaluated when this is a recursive call from within eval_apply itself
  // For now, always evaluate arguments
  bool args_evaluated = false;
  
  // Evaluate the procedure
  SCM *proc = eval_with_env(env, l->next->data);
  
  // When args are evaluated, proc might already be a procedure
  // But we still need to check
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
  
  // Evaluate or use the last argument (should be a list)
  SCM *last_arg;
  if (args_evaluated) {
    // Arguments are already evaluated
    last_arg = last_arg_node->data;
  } else {
    // Evaluate the last argument (should be a list)
    last_arg = eval_with_env(env, last_arg_node->data);
  }
  if (!is_pair(last_arg) && !is_nil(last_arg)) {
    eval_error("apply: last argument must be a list");
  }
  
  // Build the combined argument list
  // First, evaluate or use all arguments before the last one
  SCM_List args_dummy = make_list_dummy();
  SCM_List *args_tail = &args_dummy;
  
  current = args_before_last;
  while (current != last_arg_node) {
    SCM *arg_val;
    if (args_evaluated) {
      arg_val = current->data; // Already evaluated
    } else {
      arg_val = eval_with_env(env, current->data);
    }
    SCM_List *node = make_list(arg_val);
    args_tail->next = node;
    args_tail = node;
    current = current->next;
  }
  
  // Then append all elements from the last argument list
  if (is_pair(last_arg)) {
    SCM_List *last_list = cast<SCM_List>(last_arg);
    while (last_list) {
      SCM *elem;
      if (args_evaluated) {
        // Arguments are already evaluated, use directly
        elem = last_list->data;
      } else {
        // Wrap each element in quote to prevent double evaluation
        elem = scm_list2(scm_sym_quote(), last_list->data);
      }
      SCM_List *node = make_list(elem);
      args_tail->next = node;
      args_tail = node;
      last_list = last_list->next;
    }
  } else if (args_evaluated && !is_nil(last_arg)) {
    // When args are evaluated and last_arg is not a pair, it might be a single value
    // This shouldn't happen for apply, but handle it gracefully
    SCM_List *node = make_list(last_arg);
    args_tail->next = node;
  }
  
  // Apply the procedure with the combined arguments
  if (is_proc(proc)) {
    SCM_Procedure *proc_obj = cast<SCM_Procedure>(proc);
    return apply_procedure(env, proc_obj, args_dummy.next);
  } else if (is_func(proc)) {
    SCM_Function *func_obj = cast<SCM_Function>(proc);
    // Special handling for apply: if apply is being applied, we need to handle it specially
    if (func_obj->name && strcmp(func_obj->name->data, "apply") == 0) {
      // This is apply being applied - construct a call list and use eval_apply
      // args_dummy.next contains the arguments for apply: (proc arg1 arg2 ... args-list)
      // We need to mark that arguments are already evaluated
      SCM_List apply_call;
      apply_call.data = proc;
      apply_call.next = args_dummy.next;
      // Mark that arguments are already evaluated by setting args_evaluated flag
      // But we can't pass that flag, so we need to detect it in eval_apply
      // Actually, we can detect it by checking if l->data is a function (not a symbol)
      return eval_apply(env, &apply_call);
    }
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

