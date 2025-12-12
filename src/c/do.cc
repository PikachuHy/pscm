#include "pscm.h"
#include "eval.h"

// Helper function to update do loop variables
void update_do_variables(SCM_Environment *do_env, SCM_List *var_update_list) {
  auto it = var_update_list;
  while (it->next) {
    it = it->next;
    auto var_update_expr = cast<SCM_List>(it->data);
    auto var_name = cast<SCM_Symbol>(var_update_expr->data);
    auto var_update_step = var_update_expr->next->data;

    auto new_var_val = eval_with_env(do_env, var_update_step);
    if (debug_enabled) {
      SCM_DEBUG_EVAL("eval do step ... ");
      print_ast(var_update_step);
      printf(" --> ");
      print_ast(new_var_val);
      printf("\n");
    }
    scm_env_insert(do_env, var_name, new_var_val);
  }
}

// Helper function for do special form
SCM *eval_do(SCM_Environment *env, SCM_List *l) {
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

  while (var_init_it) {
    auto var_init_expr = cast<SCM_List>(var_init_it->data);
    auto var_name = cast<SCM_Symbol>(var_init_expr->data);
    auto var_init_val = eval_with_env(env, var_init_expr->next->data);
    auto var_update_step = var_init_expr->next->next->data;

    scm_env_insert(do_env, var_name, var_init_val);
    var_update_it->next = make_list(scm_list2(wrap(var_name), var_update_step));
    var_update_it = var_update_it->next;
    var_update_it->next = nullptr;
    var_init_it = var_init_it->next;
  }

  auto test_clause_list = cast<SCM_List>(test_clause);
  if (!test_clause_list) {
    eval_error("do: test clause must be a list");
    return nullptr;
  }
  auto ret = eval_with_env(do_env, car(test_clause));
  while (is_false(ret)) {
    if (body_clause) {
      eval_list_with_env(do_env, body_clause);
    }
    update_do_variables(do_env, &var_update_dummy);
    ret = eval_with_env(do_env, car(test_clause));
  }
  // Evaluate and return the result expressions from the test clause
  auto return_exprs = cdr(test_clause);
  if (is_nil(return_exprs)) {
    return scm_none();
  }
  // Evaluate all return expressions and return the last one
  auto return_list = cast<SCM_List>(return_exprs);
  SCM *last_result = scm_none();
  while (return_list) {
    last_result = eval_with_env(do_env, return_list->data);
    return_list = return_list->next;
  }
  return last_result;
}

