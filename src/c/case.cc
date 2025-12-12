#include "pscm.h"
#include "eval.h"

// Forward declaration
bool _eq(SCM *lhs, SCM *rhs);

// Helper function to check if a value is in a list of datums
// Uses eqv? for comparison (which is implemented as _eq in this codebase)
static bool value_in_datums(SCM *value, SCM_List *datums) {
  if (!datums) {
    return false;
  }
  
  for (SCM_List *it = datums; it; it = it->next) {
    if (it->data && _eq(value, it->data)) {
      return true;
    }
  }
  return false;
}

// Helper function for case special form
// Syntax: (case key ((datum1 ...) expr1 ...) ((datum2 ...) expr2 ...) ... (else expr ...))
SCM *eval_case(SCM_Environment *env, SCM_List *l) {
  assert(l->next);  // Must have at least key expression
  
  // Evaluate the key expression
  SCM *key = eval_with_env(env, l->next->data);
  
  if (debug_enabled) {
    SCM_DEBUG_EVAL("case key: ");
    print_ast(key);
    printf("\n");
  }
  
  // Process each clause
  for (SCM_List *clause_it = l->next->next; clause_it; clause_it = clause_it->next) {
    auto clause = cast<SCM_List>(clause_it->data);
    
    if (debug_enabled) {
      SCM_DEBUG_EVAL("eval case clause ");
      print_list(clause);
      printf("\n");
    }
    
    // Check for else clause
    if (is_sym_val(clause->data, "else")) {
      // Execute else clause expressions
      return eval_with_list(env, clause->next);
    }
    
    // Regular clause: (datum1 datum2 ...)
    // Check if key matches any datum in this clause
    auto datums = cast<SCM_List>(clause->data);
    
    if (value_in_datums(key, datums)) {
      // Key matches this clause, execute the expressions
      return eval_with_list(env, clause->next);
    }
    
    // Key doesn't match, continue to next clause
  }
  
  // No clause matched and no else clause
  // According to R4RS, if no clause matches, the result is unspecified
  // We'll return #f for consistency
  return scm_bool_false();
}
