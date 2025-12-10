#include "pscm.h"
#include "eval.h"

SCM *eval_define(SCM_Environment *env, SCM_List *l) {
  if (l->next && is_sym(l->next->data)) {
    // Define variable: (define var value)
    SCM_Symbol *varname = cast<SCM_Symbol>(l->next->data);
    SCM_DEBUG_EVAL("define variable %s\n", varname->data);
    auto val = eval_with_env(env, l->next->next->data);
    assert(val);
    if (is_proc(val)) {
      auto proc = cast<SCM_Procedure>(val);
      if (proc->name == nullptr) {
        proc->name = varname;
        SCM_DEBUG_EVAL("define proc from lambda\n");
      }
    }
    scm_env_insert(env, varname, val);
    return scm_none();
  }
  // Define procedure: (define (name args...) body...)
  // Handle curried form: (define ((name args1...) args2...) body...)
  SCM_DEBUG_EVAL("define a procedure");
  
  // Handle curried form: (define ((name args1...) args2...) body...)
  // Convert to: (define name (lambda (args1...) (lambda (args2...) body...)))
  SCM *key = l->next->data;
  SCM_List *val_list = l->next->next;
  
  while (is_pair(key) && !is_sym(key)) {
    // key is a list, extract proc_name and proc_args
    SCM_List *key_list = cast<SCM_List>(key);
    SCM *proc_name = key_list->data;
    SCM_List *proc_args = key_list->next;
    
    // Create lambda: (lambda proc_args val)
    // First, create the lambda symbol
    SCM_Symbol *lambda_sym = make_sym("lambda");
    
    // Create lambda list: (lambda proc_args . val_list)
    // The structure should be: (lambda . (proc_args . val_list))
    SCM_List *lambda_list = make_list(wrap(lambda_sym));
    
    // proc_args is already a list structure, so we can use it directly
    // But we need to make sure it's wrapped properly
    SCM *proc_args_wrapped = proc_args ? wrap(proc_args) : scm_nil();
    SCM_List *lambda_args_list = make_list(proc_args_wrapped);
    
    // Link lambda to args
    lambda_list->next = lambda_args_list;
    
    // Link args to body (val_list)
    SCM_List *lambda_body_tail = lambda_args_list;
    while (lambda_body_tail->next) {
      lambda_body_tail = lambda_body_tail->next;
    }
    // Copy val_list to lambda body
    if (val_list) {
      lambda_body_tail->next = val_list;
    }
    
    // Wrap lambda in a list for val: (lambda_list)
    SCM_List *new_val_list = make_list(wrap(lambda_list));
    val_list = new_val_list;
    key = proc_name;
  }
  
  if (!is_sym(key)) {
    // Regular define: (define (name args...) body...)
    SCM_List *proc_sig = cast<SCM_List>(l->next->data);
    if (!is_sym(proc_sig->data)) {
      eval_error("not supported define form");
      return nullptr;
    }
    SCM_Symbol *proc_name = cast<SCM_Symbol>(proc_sig->data);
    SCM_DEBUG_EVAL(" %s with params ", proc_name->data);
    if (debug_enabled) {
      printf("(");
      if (proc_sig->next) {
        print_ast(proc_sig->next->data);
      }
      printf(")\n");
    }
    auto proc = make_proc(proc_name, proc_sig->next, l->next->next, env);
    SCM *ret = wrap(proc);
    scm_env_insert(env, proc_name, ret);
    return ret;
  }
  
  SCM_Symbol *proc_name = cast<SCM_Symbol>(key);
  SCM_DEBUG_EVAL(" %s with params ", proc_name->data);
  if (debug_enabled) {
    printf("(");
    printf(")\n");
  }
  
  // Evaluate the lambda expression
  // val_list->data should be a wrapped list (lambda ...)
  // We need to unwrap it to get the actual list structure
  SCM *lambda_wrapped = val_list->data;
  if (is_pair(lambda_wrapped)) {
    // Unwrap to get the actual list
    SCM_List *lambda_list = cast<SCM_List>(lambda_wrapped);
    // Now evaluate it as a lambda expression
    // Create a dummy list with lambda as first element
    SCM_List *dummy_lambda_call = make_list(lambda_list->data);
    dummy_lambda_call->next = lambda_list->next;
    SCM *val = eval_lambda(env, dummy_lambda_call);
    assert(val);
    if (is_proc(val)) {
      auto proc = cast<SCM_Procedure>(val);
      if (proc->name == nullptr) {
        proc->name = proc_name;
        SCM_DEBUG_EVAL("define proc from lambda\n");
      }
    }
    scm_env_insert(env, proc_name, val);
    return scm_none();
  } else {
    // Should not happen, but handle it
    eval_error("invalid lambda expression in define");
    return nullptr;
  }
}

