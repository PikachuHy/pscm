#include "pscm.h"
#include "eval.h"

// Wrapper function for map as a builtin function
// This function is called when map is used as a value (not as a special form)
SCM *scm_c_map(SCM_List *args) {
  // This function should never be called directly
  // It's registered as a placeholder, and eval.cc will handle it specially
  eval_error("map: internal error - should be handled as special form");
  return nullptr;
}

void init_map() {
  scm_define_vararg_function("map", scm_c_map);
}

// Helper function for map special form
SCM *eval_map(SCM_Environment *env, SCM_List *l) {
  if (!l->next || !l->next->next) {
    eval_error("map: requires at least 2 arguments (procedure and list)");
  }
  
  // When map is called as a special form (from eval.cc with symbol), arguments are NOT evaluated
  // When map is called as a function value (from eval.cc with function), arguments are also NOT evaluated yet
  // We always need to evaluate arguments first
  // The only case where arguments might be pre-evaluated is when called from apply, but apply handles that
  
  // Evaluate the procedure argument
  SCM *proc = eval_with_env(env, l->next->data);
  
  // Check if proc is a procedure
  if (!is_proc(proc) && !is_func(proc)) {
    eval_error("map: first argument must be a procedure");
  }
  
  // Collect all list arguments
  SCM_List *list_args_head = l->next->next;
  int num_lists = 0;
  SCM_List *temp = list_args_head;
  while (temp) {
    num_lists++;
    temp = temp->next;
  }
  
  if (num_lists == 0) {
    eval_error("map: requires at least one list argument");
  }
  
  // Evaluate all list arguments and store them
  SCM_List **list_ptrs = new SCM_List*[num_lists];
  temp = list_args_head;
  for (int i = 0; i < num_lists; i++) {
    // Always evaluate list arguments
    SCM *list_arg = eval_with_env(env, temp->data);
    // Check if list_arg is a list
    if (!is_pair(list_arg) && !is_nil(list_arg)) {
      eval_error("map: list arguments must be lists");
    }
    list_ptrs[i] = is_nil(list_arg) ? nullptr : cast<SCM_List>(list_arg);
    temp = temp->next;
  }
  
  // Build result list by applying proc to corresponding elements
  SCM_List dummy = make_list_dummy();
  SCM_List *tail = &dummy;
  
  // Continue until the shortest list is exhausted
  bool all_non_empty = true;
  for (int i = 0; i < num_lists; i++) {
    if (!list_ptrs[i]) {
      all_non_empty = false;
      break;
    }
  }
  
  while (all_non_empty) {
    // Collect one element from each list
    // Wrap each element in quote to prevent evaluation
    SCM_List args_dummy = make_list_dummy();
    SCM_List *args_tail = &args_dummy;
    
    for (int i = 0; i < num_lists; i++) {
      // Wrap element in (quote element) to prevent evaluation
      SCM *quoted_elem = scm_list2(scm_sym_quote(), list_ptrs[i]->data);
      SCM_List *node = make_list(quoted_elem);
      args_tail->next = node;
      args_tail = node;
      list_ptrs[i] = list_ptrs[i]->next;
    }
    
    // Apply proc to collected arguments
    SCM *result;
    if (is_proc(proc)) {
      SCM_Procedure *proc_obj = cast<SCM_Procedure>(proc);
      result = apply_procedure(env, proc_obj, args_dummy.next);
    } else if (is_func(proc)) {
      SCM_Function *func_obj = cast<SCM_Function>(proc);
      SCM_List *evaled_args = eval_list_with_env(env, args_dummy.next);
      SCM_List func_call;
      func_call.data = proc;
      func_call.next = evaled_args;
      result = eval_with_func(func_obj, &func_call);
    } else {
      eval_error("map: first argument must be a procedure");
    }
    
    // Add result to result list
    SCM_List *node = make_list(result);
    tail->next = node;
    tail = node;
    
    // Check if all lists still have elements
    all_non_empty = true;
    for (int i = 0; i < num_lists; i++) {
      if (!list_ptrs[i]) {
        all_non_empty = false;
        break;
      }
    }
  }
  
  delete[] list_ptrs;
  
  if (dummy.next) {
    return wrap(dummy.next);
  }
  return scm_nil();
}

