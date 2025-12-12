#include "pscm.h"
#include "eval.h"

// Helper function to update do loop variables
// All update expressions are evaluated first (in parallel), then all variables are updated
// This implements Scheme's parallel binding semantics for do loops
void update_do_variables(SCM_Environment *do_env, SCM_List *var_update_list) {
  struct UpdatePair {
    SCM_Symbol *var_name;
    SCM *new_val;
  };
  
  // First, collect all variable names and update expressions
  // We use a vector-like approach: first count, then allocate, then process
  int count = 0;
  auto it = var_update_list;
  while (it->next) {
    count++;
    it = it->next;
  }
  
  if (count == 0) {
    return;  // No variables to update
  }
  
  // Allocate array to store variable names and new values
  UpdatePair *updates = new UpdatePair[count];
  int idx = 0;
  
  // First pass: evaluate ALL update expressions BEFORE updating any variables
  // This ensures all update expressions see the old values of all variables
  // This is critical for correct parallel binding semantics
  it = var_update_list;
  while (it->next) {
    it = it->next;
    auto list_wrapped = it->data;
    if (!is_pair(list_wrapped)) {
      eval_error("do: internal error - update list must be a pair");
      delete[] updates;
      return;
    }
    // list_wrapped is (var_name var_update_step)
    auto var_update_expr = cast<SCM_List>(list_wrapped);
    if (!var_update_expr->next) {
      eval_error("do: internal error - update list must have two elements");
      delete[] updates;
      return;
    }
    if (!is_sym(var_update_expr->data)) {
      eval_error("do: internal error - first element of update list must be a symbol");
      delete[] updates;
      return;
    }
    auto var_name = cast<SCM_Symbol>(var_update_expr->data);
    auto var_update_step = var_update_expr->next->data;

    // CRITICAL: Evaluate update expression in current environment (before any updates)
    // This ensures all update expressions see the old values of all variables
    auto new_var_val = eval_with_env(do_env, var_update_step);
    if (!new_var_val) {
      // Evaluation error occurred, clean up and return
      delete[] updates;
      return;
    }
    
    if (debug_enabled) {
      SCM_DEBUG_EVAL("eval do step ... ");
      print_ast(var_update_step);
      printf(" --> ");
      print_ast(new_var_val);
      printf("\n");
    }
    updates[idx].var_name = var_name;
    updates[idx].new_val = new_var_val;
    idx++;
  }
  
  // Second pass: update all variables with the pre-computed values
  // This happens AFTER all update expressions have been evaluated
  // This implements parallel binding: all variables are updated simultaneously
  for (int i = 0; i < count; i++) {
    scm_env_insert(do_env, updates[i].var_name, updates[i].new_val);
  }
  
  delete[] updates;
}

// Helper function for do special form
SCM *eval_do(SCM_Environment *env, SCM_List *l) {
  assert(env);
  assert(l);
  assert(l->next && l->next->next);
  auto var_init_l = cast<SCM_List>(l->next->data);
  auto test_clause = l->next->next->data;
  auto body_clause = l->next->next->next;  // May be null if no body

  if (debug_enabled) {
    SCM_DEBUG_EVAL("eval do\n");
    printf("var: ");
    print_list(var_init_l);
    printf("\n");
    printf("test: ");
    print_ast(test_clause);
    printf("\n");
    printf("cmd: ");
    print_list(body_clause);
    printf("\n");
  }
  auto do_env = make_env(env);

  auto var_init_it = var_init_l;
  SCM_List var_update_dummy = make_list_dummy();
  auto var_update_it = &var_update_dummy;

  // Process variable bindings: (var init [step])
  while (var_init_it) {
    if (!is_pair(var_init_it->data)) {
      eval_error("do: variable initialization must be a list");
      return nullptr;
    }
    
    auto var_init_expr = cast<SCM_List>(var_init_it->data);
    
    if (!is_sym(var_init_expr->data)) {
      eval_error("do: variable name must be a symbol");
      return nullptr;
    }
    
    auto var_name = cast<SCM_Symbol>(var_init_expr->data);
    
    // Evaluate initial value in the parent environment
    if (!var_init_expr->next) {
      eval_error("do: variable binding must have an initial value");
      return nullptr;
    }
    auto var_init_val = eval_with_env(env, var_init_expr->next->data);
    if (!var_init_val) {
      return nullptr;  // Evaluation error
    }
    
    // Check for optional update step
    SCM *var_update_step = nullptr;
    if (var_init_expr->next->next) {
      var_update_step = var_init_expr->next->next->data;
    }

    // Insert variable into do environment
    scm_env_insert(do_env, var_name, var_init_val);
    
    // Store update step if provided
    if (var_update_step) {
      // Store as a list: (var_name var_update_step)
      SCM *var_name_wrapped = wrap(var_name);
      SCM *pair = scm_list2(var_name_wrapped, var_update_step);
      var_update_it->next = make_list(pair);
      var_update_it = var_update_it->next;
      var_update_it->next = nullptr;
    }
    var_init_it = var_init_it->next;
  }

  auto test_clause_list = cast<SCM_List>(test_clause);
  if (!test_clause_list) {
    eval_error("do: test clause must be a list");
    return nullptr;
  }
  
  if (debug_enabled) {
    SCM_DEBUG_EVAL("do: test_clause_list: ");
    print_list(test_clause_list);
    printf("\n");
  }
  
  // Evaluate initial test condition
  // Use test_clause (SCM*) for car/cdr, not test_clause_list (SCM_List*)
  auto test_expr = car(test_clause);
  auto ret = eval_with_env(do_env, test_expr);
  if (!ret) {
    return nullptr;  // Evaluation error
  }
  
  if (debug_enabled) {
    SCM_DEBUG_EVAL("do: initial test result: ");
    print_ast(ret);
    printf("\n");
  }
  
  // Main loop: continue while test condition is false
  int loop_count = 0;
  while (is_false(ret)) {
    if (debug_enabled) {
      SCM_DEBUG_EVAL("do: loop iteration %d\n", loop_count++);
    }
    
    // Execute body if present
    if (body_clause) {
      if (debug_enabled) {
        SCM_DEBUG_EVAL("do: executing body: ");
        print_list(body_clause);
        printf("\n");
      }
      eval_list_with_env(do_env, body_clause);
    }
    
    // Update all variables (parallel binding)
    update_do_variables(do_env, &var_update_dummy);
    
    // Re-evaluate test condition
    ret = eval_with_env(do_env, car(test_clause));
    if (!ret) {
      return nullptr;  // Evaluation error
    }
    
    if (debug_enabled) {
      SCM_DEBUG_EVAL("do: test result after update: ");
      print_ast(ret);
      printf("\n");
    }
  }
  
  // Loop exited: test condition is true
  // Evaluate and return the result expressions from the test clause
  auto return_exprs = cdr(test_clause);
  if (is_nil(return_exprs)) {
    return scm_none();  // No return expressions, return unspecified value
  }
  
  // Evaluate all return expressions and return the last one
  auto return_list = cast<SCM_List>(return_exprs);
  SCM *last_result = scm_none();
  while (return_list) {
    last_result = eval_with_env(do_env, return_list->data);
    if (!last_result) {
      return nullptr;  // Evaluation error
    }
    return_list = return_list->next;
  }
  return last_result;
}

