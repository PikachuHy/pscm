#include "pscm.h"
#include "eval.h"
#include <type_traits>
extern bool _number_eq(SCM *lhs, SCM *rhs);
bool _eq(SCM *lhs, SCM *rhs);

bool _sym_eq(SCM *lhs, SCM *rhs) {
  assert(lhs);
  assert(rhs);
  assert(is_sym(lhs) || is_str(lhs));
  assert(is_sym(rhs) || is_str(rhs));
  
  // Handle symbols
  if (is_sym(lhs) && is_sym(rhs)) {
    auto sym1 = cast<SCM_Symbol>(lhs);
    auto sym2 = cast<SCM_Symbol>(rhs);
    return strcmp(sym1->data, sym2->data) == 0;
  }
  
  // Handle strings
  if (is_str(lhs) && is_str(rhs)) {
    auto str1 = cast<SCM_String>(lhs);
    auto str2 = cast<SCM_String>(rhs);
    if (str1->len != str2->len) {
      return false;
    }
    return strncmp(str1->data, str2->data, str1->len) == 0;
  }
  
  // Mixed types (sym and str) are not equal
  return false;
}

bool _list_eq(SCM *lhs, SCM *rhs) {
  auto l1 = cast<SCM_List>(lhs);
  auto l2 = cast<SCM_List>(rhs);
  while (l1 && l2) {
    if (!_eq(l1->data, l2->data)) {
      return false;
    }
    l1 = l1->next;
    l2 = l2->next;
  }
  if (l1 || l2) {
    return false;
  }
  return true;
}

bool _vector_eq(SCM *lhs, SCM *rhs) {
  auto v1 = cast<SCM_Vector>(lhs);
  auto v2 = cast<SCM_Vector>(rhs);
  
  // Vectors must have the same length
  if (v1->length != v2->length) {
    return false;
  }
  
  // Compare each element recursively
  for (size_t i = 0; i < v1->length; i++) {
    if (!_eq(v1->elements[i], v2->elements[i])) {
      return false;
    }
  }
  
  return true;
}

bool _eq(SCM *lhs, SCM *rhs) {
  assert(lhs);
  assert(rhs);
  // Special case: allow NUM <-> FLOAT comparison
  if ((lhs->type == SCM::NUM && rhs->type == SCM::FLOAT) ||
      (lhs->type == SCM::FLOAT && rhs->type == SCM::NUM)) {
    return _number_eq(lhs, rhs);
  }
  
  if (lhs->type != rhs->type) {
    return false;
  }

  switch (lhs->type) {
  case SCM::NONE:
  case SCM::NIL:
    return true;
  case SCM::LIST:
    return _list_eq(lhs, rhs);
  case SCM::PROC:
  case SCM::CONT:
  case SCM::FUNC:
    return lhs == rhs;
  case SCM::NUM:
  case SCM::FLOAT:
    return _number_eq(lhs, rhs);
  case SCM::CHAR:
    // Character comparison
    if (rhs->type == SCM::CHAR) {
      char ch_lhs = ptr_to_char(lhs->value);
      char ch_rhs = ptr_to_char(rhs->value);
      return ch_lhs == ch_rhs;
    }
    return false;
  case SCM::BOOL:
    return is_true(lhs) == is_true(rhs);
  case SCM::SYM:
  case SCM::STR:
    return _sym_eq(lhs, rhs);
  case SCM::VECTOR:
    return _vector_eq(lhs, rhs);
  case SCM::MACRO:
    // Macros are compared by reference (same as procedures)
    return lhs == rhs;
  case SCM::HASH_TABLE:
    // Hash tables are compared by reference
    return lhs == rhs;
  case SCM::RATIO:
    // Ratios are compared numerically
    return _number_eq(lhs, rhs);
  default:
    SCM_ERROR_EVAL("unsupported scheme type %d", lhs->type);
    eval_error("unsupported scheme type %d", lhs->type);
    return false;
  }
}

SCM *scm_c_is_eq(SCM *lhs, SCM *rhs) {
  if (_eq(lhs, rhs)) {
    return scm_bool_true();
  }
  return scm_bool_false();
}

void init_eq() {
  scm_define_function("eq?", 2, 0, 0, scm_c_is_eq);
  scm_define_function("eqv?", 2, 0, 0, scm_c_is_eq);
  scm_define_function("equal?", 2, 0, 0, scm_c_is_eq);
}