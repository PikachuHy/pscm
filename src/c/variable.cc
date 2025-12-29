#include "pscm.h"
#include "eval.h"
#include <cstring>

// Make a variable initialized to a value
SCM *scm_make_variable(SCM *init) {
  auto var = new SCM_Variable();
  var->value = init;
  return wrap(var);
}

// Make an undefined variable
SCM *scm_make_undefined_variable(void) {
  auto var = new SCM_Variable();
  var->value = nullptr;  // nullptr means unbound
  return wrap(var);
}

// Check if an object is a variable
SCM *scm_variable_p(SCM *obj) {
  return bool_to_scm(is_variable(obj));
}

// Get the value of a variable
SCM *scm_variable_ref(SCM *var) {
  if (!is_variable(var)) {
    eval_error("variable-ref: expected variable object");
  }
  SCM_Variable *v = cast<SCM_Variable>(var);
  if (!v->value) {
    eval_error("variable-ref: variable is unbound");
  }
  return v->value;
}

// Set the value of a variable
SCM *scm_variable_set_x(SCM *var, SCM *val) {
  if (!is_variable(var)) {
    eval_error("variable-set!: expected variable object");
  }
  SCM_Variable *v = cast<SCM_Variable>(var);
  v->value = val;
  return scm_none();  // Return unspecified value
}

// Check if a variable is bound
SCM *scm_variable_bound_p(SCM *var) {
  if (!is_variable(var)) {
    eval_error("variable-bound?: expected variable object");
  }
  SCM_Variable *v = cast<SCM_Variable>(var);
  return bool_to_scm(v->value != nullptr);
}

// Look up a symbol in the current module and return a variable object
// Similar to Guile's scm_c_lookup
// Throws an error if the variable is not found
SCM *scm_c_lookup(const char *name) {
  // Create symbol from name
  SCM_Symbol *sym = make_sym(name);
  
  // Look up in current environment/module
  SCM *value = scm_env_search(&g_env, sym);
  
  if (!value) {
    // Not found - throw error (like Guile)
    eval_error("scm_c_lookup: unbound variable: %s", name);
  }
  
  // Found - create a variable object wrapping the value
  return scm_make_variable(value);
}

// Look up a symbol (SCM*) in the current module and return a variable object
SCM *scm_lookup(SCM *sym) {
  if (!is_sym(sym)) {
    eval_error("scm_lookup: expected symbol");
  }
  
  SCM_Symbol *sym_obj = cast<SCM_Symbol>(sym);
  
  // Look up in current environment/module
  SCM *value = scm_env_search(&g_env, sym_obj);
  
  if (!value) {
    eval_error("scm_lookup: unbound variable: %s", sym_obj->data);
  }
  
  // Found - create a variable object wrapping the value
  return scm_make_variable(value);
}

// Initialize variable module
void init_variable() {
  scm_define_function("make-variable", 1, 0, 0, scm_make_variable);
  scm_define_function("make-undefined-variable", 0, 0, 0, scm_make_undefined_variable);
  scm_define_function("variable?", 1, 0, 0, scm_variable_p);
  scm_define_function("variable-ref", 1, 0, 0, scm_variable_ref);
  scm_define_function("variable-set!", 2, 0, 0, scm_variable_set_x);
  scm_define_function("variable-bound?", 1, 0, 0, scm_variable_bound_p);
}

