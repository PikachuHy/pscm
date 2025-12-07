#include "pscm.h"
#include "eval.h"

// Helper function to check if a symbol is literal (not bound in environment)
// Similar to Guile's literal_p function
static bool is_literal(SCM_Environment *env, SCM_Symbol *sym) {
  return scm_env_exist(env, sym) == nullptr;
}

// Helper function for cond special form
SCM *eval_cond(SCM_Environment *env, SCM_List *l, SCM **ast) {
  assert(l->next);
  for (auto it = l->next; it; it = it->next) {
    auto clause = cast<SCM_List>(it->data);
    if (debug_enabled) {
      SCM_DEBUG_EVAL("eval cond clause ");
      print_list(clause);
      printf("\n");
    }
    if (is_sym_val(clause->data, "else")) {
      return eval_with_list(env, clause->next);
    }
    auto pred = eval_with_env(env, clause->data);
    if (is_bool(pred) && is_false(pred)) {
      continue;
    }
    if (!clause->next) {
      return scm_bool_true();
    }
    // Check for => syntax BEFORE evaluating clause->next->data
    // Similar to Guile's approach: check if => is a literal (not bound)
    // This allows => to be bound to a value and still work as cond syntax
    if (is_sym(clause->next->data) && is_sym_val(clause->next->data, "=>")) {
      SCM_Symbol *arrow_sym = cast<SCM_Symbol>(clause->next->data);
      // Check if => is literal (not bound in environment), similar to Guile's literal_p
      if (is_literal(env, arrow_sym)) {
        assert(clause->next->next);
        // First evaluate the procedure (clause->next->next->data) to get the actual procedure
        // Then call it with pred as argument
        // Build: (proc (quote pred))
        // Note: eval_with_env will handle quote expressions and return the quoted value
        // If the result is a symbol, we need to look it up in the environment to get the procedure
        SCM *proc = eval_with_env(env, clause->next->next->data);
        // If proc is a symbol (from quote), look it up in the environment
        // This handles the case where 'ok is evaluated to symbol ok, then we look up ok
        // But if the symbol is not found, it means we're trying to use a quoted symbol as a procedure
        // In that case, we should just return the symbol itself (not call it)
        if (is_sym(proc)) {
          SCM *val = scm_env_exist(env, cast<SCM_Symbol>(proc));
          if (val) {
            proc = val;
          } else {
            // Symbol not found in environment - this means 'ok was just a quoted symbol
            // Return the symbol itself (not as a procedure call)
            return proc;
          }
        }
        *ast = scm_list2(proc, scm_list2(scm_sym_quote(), pred));
        return nullptr; // Signal to continue evaluation
      }
      // If => is bound to a value, treat it as a regular expression (not => syntax)
      // Fall through to normal evaluation
    }
    // Normal clause: evaluate the expressions in sequence
    return eval_with_list(env, clause->next);
  }
  return scm_none();
}

