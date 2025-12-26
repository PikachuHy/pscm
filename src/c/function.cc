#include "pscm.h"
#include "eval.h"

SCM *eval_with_func_0(SCM_Function *func) {
  typedef SCM *(*func_0)();
  auto f = (func_0)func->func_ptr;
  return f();
}

SCM *eval_with_func_1(SCM_Function *func, SCM *arg1) {
  typedef SCM *(*func_1)(SCM *);
  auto f = (func_1)func->func_ptr;
  return f(arg1);
}

SCM *eval_with_func_2(SCM_Function *func, SCM *arg1, SCM *arg2) {
  typedef SCM *(*func_2)(SCM *, SCM *);
  auto f = (func_2)func->func_ptr;
  return f(arg1, arg2);
}

SCM *eval_with_func_3(SCM_Function *func, SCM *arg1, SCM *arg2, SCM *arg3) {
  typedef SCM *(*func_3)(SCM *, SCM *, SCM *);
  auto f = (func_3)func->func_ptr;
  return f(arg1, arg2, arg3);
}

SCM *eval_with_func(SCM_Function *func, SCM_List *l) {
  if (debug_enabled) {
    SCM_DEBUG_EVAL("eval func ");
    printf("%s with ", func->name->data);
    print_list(l->next);
    printf("\n");
  }
  if (func->n_args == 0) {
    return eval_with_func_0(func);
  }
  if (func->n_args == 1) {
    assert(l->next);
    return eval_with_func_1(func, l->next->data);
  }
  if (func->n_args == 2) {
    assert(l->next && l->next->next);
    return eval_with_func_2(func, l->next->data, l->next->next->data);
  }
  if (func->n_args == 3) {
    assert(l->next && l->next->next && l->next->next->next);
    return eval_with_func_3(func, l->next->data, l->next->next->data, l->next->next->next->data);
  }
  if (func->n_args == -1 && func->generic) {
    return reduce(
        [func](SCM *lhs, SCM *rhs) {
          return eval_with_func_2(func, lhs, rhs);
        },
        func->generic, l->next);
  }
  if (func->n_args == -2) {
    // Variable argument function (like list)
    // Special handling for apply, map, and map-in-order: they need access to environment
    // But we can't access env here, so these should be handled in eval.cc
    // If they reach here, it's an error
    if (func->name) {
      if (strcmp(func->name->data, "apply") == 0) {
        eval_error("apply: internal error - should be handled as special form");
        return nullptr;
      }
      if (strcmp(func->name->data, "map") == 0) {
        eval_error("map: internal error - should be handled as special form");
        return nullptr;
      }
      if (strcmp(func->name->data, "map-in-order") == 0) {
        eval_error("map-in-order: internal error - should be handled as special form");
        return nullptr;
      }
    }
    typedef SCM *(*func_var)(SCM_List *);
    auto f = (func_var)func->func_ptr;
    return f(l->next);
  }
  eval_error("not supported function: %s", func->name->data);
  return nullptr; // Never reached, but satisfies compiler
}

