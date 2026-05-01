#include "pscm_api.h"
#include <cassert>
#include <cstdio>

// Callback for scm_c_call_with_current_module test
SCM *define_in_module_cb(void *data) {
  (void)data;
  // Current module is set to our test module by scm_c_call_with_current_module
  pscm_eval_string("(define y-from-cb 123)");
  return pscm_eval_string("y-from-cb");
}

int main() {
  pscm_init();

  // Test 1: scm_c_define_module creates a new module
  SCM *mod = scm_c_define_module("(my-test-module)", nullptr, nullptr);
  assert(mod != nullptr);
  assert(is_module(mod));

  // Test 2: scm_c_module_define in a specific module, then look it up
  scm_c_module_define(mod, "hello", pscm_eval_string("\"world\""));
  SCM *var = scm_c_module_lookup(mod, "hello");
  assert(var != nullptr);
  assert(is_variable(var));
  SCM *val = scm_variable_ref(var);
  assert(val != nullptr);
  assert(is_str(val));

  // Test 3: scm_c_call_with_current_module
  // Temporarily switch to module, define a variable, and verify it's accessible
  SCM *cb_result = scm_c_call_with_current_module(mod, define_in_module_cb, nullptr);
  assert(cb_result != nullptr);

  // Verify the variable defined in Test 3 persists in the module
  SCM *old = scm_set_current_module(mod);
  SCM *r = pscm_eval_string("y-from-cb");
  assert(r && is_num(r));

  // Verify we can also look up via C API
  SCM *var2 = scm_c_module_lookup(mod, "y-from-cb");
  assert(var2 != nullptr);
  assert(is_variable(var2));
  SCM *val2 = scm_variable_ref(var2);
  assert(val2 != nullptr);
  assert(is_num(val2));

  // Restore current module
  scm_set_current_module(old);

  printf("c_api_module: PASS\n");
  return 0;
}
