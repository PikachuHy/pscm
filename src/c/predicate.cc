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

SCM *scm_c_is_list(SCM *arg) {
  assert(arg);
  // nil is a proper list
  if (is_nil(arg)) {
    return scm_bool_true();
  }
  // If not a pair, it's not a list
  if (!is_pair(arg)) {
    return scm_bool_false();
  }
  // Traverse the list to check if it's a proper list
  // A proper list ends with a node where next=nullptr and is_dotted=false
  // (cdr of the last node returns nil)
  SCM_List *l = cast<SCM_List>(arg);
  // Use a simple cycle detection: limit traversal count
  int count = 0;
  const int MAX_COUNT = 10000;  // Reasonable limit for list length
  
  while (l) {
    if (count++ > MAX_COUNT) {
      // Likely a circular list, return #f
      return scm_bool_false();
    }
    // Check if this node is a dotted pair marker
    if (l->is_dotted) {
      // This is a dotted pair, not a proper list
      return scm_bool_false();
    }
    // If next is nullptr, we've reached the end of a proper list
    if (!l->next) {
      // Proper list ends with next=nullptr and is_dotted=false
      return scm_bool_true();
    }
    // Move to next node
    l = l->next;
  }
  // Should not reach here, but if we do, it's not a proper list
  return scm_bool_false();
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
  scm_define_function("list?", 1, 0, 0, scm_c_is_list);
  scm_define_function("number?", 1, 0, 0, scm_c_is_number);
  scm_define_function("string?", 1, 0, 0, scm_c_is_string);
  // char? is registered in init_char()
  scm_define_function("symbol?", 1, 0, 0, scm_c_is_symbol);
  scm_define_function("vector?", 1, 0, 0, scm_c_is_vector);
  scm_define_function("not", 1, 0, 0, scm_c_not);
  scm_define_function("noop", 0, 0, 0, scm_c_noop);
}

