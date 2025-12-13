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

SCM *scm_c_is_number(SCM *arg) {
  return scm_c_type_check(arg, [](SCM *a) { return is_num(a) || is_float(a); });
}

SCM *scm_c_is_string(SCM *arg) {
  return scm_c_type_check(arg, is_str);
}

SCM *scm_c_is_symbol(SCM *arg) {
  return scm_c_type_check(arg, is_sym);
}

SCM *scm_c_is_vector(SCM *arg) {
  return scm_c_type_check(arg, [](SCM *a) { return a->type == SCM::VECTOR; });
}

SCM *scm_c_not(SCM *arg) {
  // In Scheme, not returns #t if arg is #f, #f otherwise
  // Only #f is falsy in Scheme, everything else (including nil) is truthy
  if (is_falsy(arg)) {
    return scm_bool_true();
  }
  return scm_bool_false();
}

SCM *scm_c_noop() {
  return scm_bool_false();
}

void init_predicate() {
  scm_define_function("procedure?", 1, 0, 0, scm_c_is_procedure);
  scm_define_function("boolean?", 1, 0, 0, scm_c_is_boolean);
  scm_define_function("null?", 1, 0, 0, scm_c_is_null);
  scm_define_function("pair?", 1, 0, 0, scm_c_is_pair);
  scm_define_function("number?", 1, 0, 0, scm_c_is_number);
  scm_define_function("string?", 1, 0, 0, scm_c_is_string);
  // char? is registered in init_char()
  scm_define_function("symbol?", 1, 0, 0, scm_c_is_symbol);
  scm_define_function("vector?", 1, 0, 0, scm_c_is_vector);
  scm_define_function("not", 1, 0, 0, scm_c_not);
  scm_define_function("noop", 0, 0, 0, scm_c_noop);
}

