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

// Forward declaration
static bool _equal_recursive(SCM *lhs, SCM *rhs);

// Helper function for equal? to compare lists recursively
// This function handles both proper lists and dotted pairs
static bool _list_equal(SCM *lhs, SCM *rhs) {
  auto l1 = cast<SCM_List>(lhs);
  auto l2 = cast<SCM_List>(rhs);
  
  while (l1 && l2) {
    // For equal?, recursively compare elements using equal?
    if (!_equal_recursive(l1->data, l2->data)) {
      return false;
    }
    
    // Check if either is a dotted pair (is_dotted flag on next node)
    bool l1_dotted = (l1->next && l1->next->is_dotted);
    bool l2_dotted = (l2->next && l2->next->is_dotted);
    
    if (l1_dotted || l2_dotted) {
      // Handle dotted pairs
      if (l1_dotted && l2_dotted) {
        // Both are dotted pairs - compare their cdrs
        return _equal_recursive(l1->next->data, l2->next->data);
      }
      
      // One is dotted, one is not - need to check if they're semantically equal
      // For example: (a . (b . (c . ()))) should equal (a b c)
      // We need to "unfold" the dotted pair and compare
      SCM *l1_cdr = l1_dotted ? l1->next->data : (l1->next ? wrap(l1->next) : scm_nil());
      SCM *l2_cdr = l2_dotted ? l2->next->data : (l2->next ? wrap(l2->next) : scm_nil());
      
      // If one cdr is a pair and the other is a list, recursively compare
      if (is_pair(l1_cdr) && is_pair(l2_cdr)) {
        return _list_equal(l1_cdr, l2_cdr);
      }
      
      // Otherwise, compare directly
      return _equal_recursive(l1_cdr, l2_cdr);
    }
    
    // Move to next node
    l1 = l1->next;
    l2 = l2->next;
    
    // Check if we've reached the end
    if (!l1 && !l2) {
      return true;
    }
    if (!l1 || !l2) {
      return false;
    }
  }
  
  // Both should be nil at this point
  return (!l1 && !l2);
}

// Legacy function for backward compatibility (used by _eq)
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

// Helper function for equal? to compare vectors recursively
static bool _vector_equal(SCM *lhs, SCM *rhs) {
  auto v1 = cast<SCM_Vector>(lhs);
  auto v2 = cast<SCM_Vector>(rhs);
  
  // Vectors must have the same length
  if (v1->length != v2->length) {
    return false;
  }
  
  // Compare each element recursively using equal?
  for (size_t i = 0; i < v1->length; i++) {
    if (!_equal_recursive(v1->elements[i], v2->elements[i])) {
      return false;
    }
  }
  
  return true;
}

// Legacy function for backward compatibility (used by _eq)
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
    // For backward compatibility with existing tests, use deep comparison for lists
    // In strict R4RS, eq? should use pointer equality (lhs == rhs) for lists
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

// eq?: Pointer equality (identity)
// According to R4RS, eq? uses pointer equality for lists and pairs
SCM *scm_c_is_eq(SCM *lhs, SCM *rhs) {
  // Pointer equality
  if (lhs == rhs) {
    return scm_bool_true();
  }
  
  // Special cases for immediate values
  if (lhs->type != rhs->type) {
    // Special case: allow NUM <-> FLOAT comparison
    if ((lhs->type == SCM::NUM && rhs->type == SCM::FLOAT) ||
        (lhs->type == SCM::FLOAT && rhs->type == SCM::NUM)) {
      return _number_eq(lhs, rhs) ? scm_bool_true() : scm_bool_false();
    }
    return scm_bool_false();
  }
  
  switch (lhs->type) {
  case SCM::NONE:
  case SCM::NIL:
    return scm_bool_true();
  case SCM::LIST:
  case SCM::VECTOR:
  case SCM::PROC:
  case SCM::CONT:
  case SCM::FUNC:
  case SCM::MACRO:
  case SCM::HASH_TABLE:
  case SCM::PORT:
    // For compound types, eq? uses pointer equality
    return scm_bool_false();
  case SCM::NUM:
  case SCM::FLOAT:
  case SCM::RATIO:
    // For numbers, compare by value
    return _number_eq(lhs, rhs) ? scm_bool_true() : scm_bool_false();
  case SCM::CHAR:
    {
      char ch_lhs = ptr_to_char(lhs->value);
      char ch_rhs = ptr_to_char(rhs->value);
      return (ch_lhs == ch_rhs) ? scm_bool_true() : scm_bool_false();
    }
  case SCM::BOOL:
    return (is_true(lhs) == is_true(rhs)) ? scm_bool_true() : scm_bool_false();
  case SCM::SYM:
  case SCM::STR:
    // For symbols and strings, compare by content (they are interned)
    return _sym_eq(lhs, rhs) ? scm_bool_true() : scm_bool_false();
  default:
    return scm_bool_false();
  }
}

// eqv?: Value equality (but pointer equality for lists/vectors/pairs)
// According to R4RS, eqv? uses pointer equality for lists and pairs (same as eq?)
SCM *scm_c_is_eqv(SCM *lhs, SCM *rhs) {
  // Pointer equality
  if (lhs == rhs) {
    return scm_bool_true();
  }
  
  // Special cases for immediate values
  if (lhs->type != rhs->type) {
    // Special case: allow NUM <-> FLOAT comparison
    if ((lhs->type == SCM::NUM && rhs->type == SCM::FLOAT) ||
        (lhs->type == SCM::FLOAT && rhs->type == SCM::NUM)) {
      return _number_eq(lhs, rhs) ? scm_bool_true() : scm_bool_false();
    }
    return scm_bool_false();
  }
  
  switch (lhs->type) {
  case SCM::NONE:
  case SCM::NIL:
    return scm_bool_true();
  case SCM::LIST:
  case SCM::VECTOR:
  case SCM::PROC:
  case SCM::CONT:
  case SCM::FUNC:
  case SCM::MACRO:
  case SCM::HASH_TABLE:
  case SCM::PORT:
    // For compound types, eqv? uses pointer equality (same as eq?)
    return scm_bool_false();
  case SCM::NUM:
  case SCM::FLOAT:
  case SCM::RATIO:
    // For numbers, compare by value
    return _number_eq(lhs, rhs) ? scm_bool_true() : scm_bool_false();
  case SCM::CHAR:
    {
      char ch_lhs = ptr_to_char(lhs->value);
      char ch_rhs = ptr_to_char(rhs->value);
      return (ch_lhs == ch_rhs) ? scm_bool_true() : scm_bool_false();
    }
  case SCM::BOOL:
    return (is_true(lhs) == is_true(rhs)) ? scm_bool_true() : scm_bool_false();
  case SCM::SYM:
  case SCM::STR:
    // For symbols and strings, compare by content (they are interned)
    return _sym_eq(lhs, rhs) ? scm_bool_true() : scm_bool_false();
  default:
    return scm_bool_false();
  }
}

// Recursive helper for equal? to avoid infinite recursion
static bool _equal_recursive(SCM *lhs, SCM *rhs) {
  if (lhs == rhs) {
    return true;
  }
  
  if (lhs->type != rhs->type) {
    // Special case: allow NUM <-> FLOAT comparison
    if ((lhs->type == SCM::NUM && rhs->type == SCM::FLOAT) ||
        (lhs->type == SCM::FLOAT && rhs->type == SCM::NUM)) {
      return _number_eq(lhs, rhs);
    }
    return false;
  }
  
  switch (lhs->type) {
  case SCM::NONE:
  case SCM::NIL:
    return true;
  case SCM::LIST:
    return _list_equal(lhs, rhs);
  case SCM::VECTOR:
    return _vector_equal(lhs, rhs);
  case SCM::PROC:
  case SCM::CONT:
  case SCM::FUNC:
  case SCM::MACRO:
  case SCM::HASH_TABLE:
  case SCM::PORT:
    // For compound types, equal? uses pointer equality (same as eq?)
    return false;
  case SCM::NUM:
  case SCM::FLOAT:
  case SCM::RATIO:
    return _number_eq(lhs, rhs);
  case SCM::CHAR:
    {
      char ch_lhs = ptr_to_char(lhs->value);
      char ch_rhs = ptr_to_char(rhs->value);
      return (ch_lhs == ch_rhs);
    }
  case SCM::BOOL:
    return (is_true(lhs) == is_true(rhs));
  case SCM::SYM:
  case SCM::STR:
    return _sym_eq(lhs, rhs);
  default:
    return false;
  }
}

// equal?: Deep value equality
SCM *scm_c_is_equal(SCM *lhs, SCM *rhs) {
  return _equal_recursive(lhs, rhs) ? scm_bool_true() : scm_bool_false();
}

void init_eq() {
  scm_define_function("eq?", 2, 0, 0, scm_c_is_eq);
  scm_define_function("eqv?", 2, 0, 0, scm_c_is_eqv);
  scm_define_function("equal?", 2, 0, 0, scm_c_is_equal);
}