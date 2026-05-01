#include "pscm_api.h"
#include <cassert>
#include <cstdio>

int main() {
  pscm_init();

  // Test 1: define a number from C
  pscm_c_define("my-num", pscm_eval_string("42"));
  SCM *r = pscm_eval_string("my-num");
  assert(r && is_num(r));

  // Test 2: overwrite existing variable
  pscm_c_define("my-num", pscm_eval_string("99"));
  r = pscm_eval_string("(+ my-num 1)");
  assert(r && is_num(r));

  // Test 3: define a procedure from C and call it from Scheme
  pscm_c_define("cube", pscm_eval_string("(lambda (x) (* x x x))"));
  r = pscm_eval_string("(cube 3)");
  assert(r && is_num(r));

  printf("c_api_basic: PASS\n");
  return 0;
}
