#include "pscm.h"
#include "eval.h"

// values: Return its arguments as a list
// This is used by call-with-values to return multiple values
SCM *scm_c_values(SCM_List *args) {
  // Return the arguments as a list
  if (!args) {
    return scm_nil();
  }
  // args is a list where each node's data is an argument
  // We need to extract the data from each node to build the result list
  SCM_List dummy = make_list_dummy();
  SCM_List *tail = &dummy;
  SCM_List *current = args;
  while (current) {
    tail->next = make_list(current->data);
    tail = tail->next;
    current = current->next;
  }
  return dummy.next ? wrap(dummy.next) : scm_nil();
}

// Helper function for call-with-values special form
// (call-with-values producer consumer)
// Calls producer, gets multiple values (as a list), then applies consumer to those values
SCM *eval_call_with_values(SCM_Environment *env, SCM_List *l) {
  if (!l->next || !l->next->next) {
    eval_error("call-with-values: requires 2 arguments (producer and consumer)");
    return nullptr;
  }
  
  // Evaluate producer
  SCM *producer = eval_with_env(env, l->next->data);
  if (!is_proc(producer) && !is_func(producer)) {
    eval_error("call-with-values: first argument must be a procedure");
    return nullptr;
  }
  
  // Call producer to get values
  SCM *values_result;
  if (is_proc(producer)) {
    SCM_Procedure *proc = cast<SCM_Procedure>(producer);
    values_result = apply_procedure(env, proc, nullptr);
  } else {
    // Function - need to call it
    SCM_Function *func = cast<SCM_Function>(producer);
    SCM_List func_call;
    func_call.data = producer;
    func_call.next = nullptr;
    values_result = eval_with_func(func, &func_call);
  }
  
  // values_result should be a list of values
  // If it's not a pair, wrap it in a list
  SCM_List *values_list;
  if (is_pair(values_result)) {
    values_list = cast<SCM_List>(values_result);
  } else if (is_nil(values_result)) {
    values_list = nullptr;
  } else {
    // Single value, wrap in list
    values_list = make_list(values_result);
  }
  
  // Get consumer
  SCM *consumer = eval_with_env(env, l->next->next->data);
  if (!is_proc(consumer) && !is_func(consumer)) {
    eval_error("call-with-values: second argument must be a procedure");
    return nullptr;
  }
  
  // Apply consumer to the values
  // Build argument list from values_list
  if (is_proc(consumer)) {
    SCM_Procedure *proc = cast<SCM_Procedure>(consumer);
    return apply_procedure(env, proc, values_list);
  } else {
    // Function - need to call it with the values
    SCM_Function *func = cast<SCM_Function>(consumer);
    SCM_List func_call;
    func_call.data = consumer;
    func_call.next = values_list;
    return eval_with_func(func, &func_call);
  }
}

void init_values() {
  scm_define_vararg_function("values", scm_c_values);
}
