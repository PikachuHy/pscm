#include "pscm.h"
#include <assert.h>

// Helper function for type checking predicates
template <typename Predicate>
SCM *scm_c_type_check(SCM *arg, Predicate pred) {
  assert(arg);
  return pred(arg) ? scm_bool_true() : scm_bool_false();
}

SCM *scm_c_is_procedure(SCM *arg) {
  return scm_c_type_check(arg, [](SCM *a) { return is_proc(a) || is_cont(a) || is_func(a); });
}

SCM *scm_c_is_boolean(SCM *arg) {
  return scm_c_type_check(arg, is_bool);
}

SCM *scm_c_is_null(SCM *arg) {
  return scm_c_type_check(arg, is_nil);
}

SCM *scm_c_is_pair(SCM *arg) {
  return scm_c_type_check(arg, is_pair);
}

SCM *scm_c_not(SCM *arg) {
  if (is_false(arg) || is_nil(arg)) {
    return scm_bool_true();
  }
  return scm_bool_false();
}

void init_predicate() {
  scm_define_function("procedure?", 1, 0, 0, scm_c_is_procedure);
  scm_define_function("boolean?", 1, 0, 0, scm_c_is_boolean);
  scm_define_function("null?", 1, 0, 0, scm_c_is_null);
  scm_define_function("pair?", 1, 0, 0, scm_c_is_pair);
  scm_define_function("not", 1, 0, 0, scm_c_not);
}

