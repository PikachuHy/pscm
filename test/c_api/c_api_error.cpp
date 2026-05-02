#include "pscm_api.h"
#include <cassert>
#include <cstdio>
#include <cstring>

int main() {
  pscm_init();

  // Test 1: parse error returns NULL
  {
    SCM *result = pscm_eval_string("(define");
    assert(result == nullptr);
    const char *key = pscm_get_last_error_key();
    const char *msg = pscm_get_last_error_message();
    assert(key != nullptr);
    assert(strcmp(key, "error") == 0);
    assert(msg != nullptr);
    printf("PASS: parse error -> key='%s', msg='%s'\n", key, msg);
  }

  // Test 2: unbound variable returns NULL
  {
    SCM *result = pscm_eval_string("undefined-variable");
    assert(result == nullptr);
    const char *msg = pscm_get_last_error_message();
    assert(msg != nullptr);
    printf("PASS: unbound variable -> msg='%s'\n", msg);
  }

  // Test 3: wrong number of args returns NULL
  {
    SCM *result = pscm_eval_string("(car 1 2)");
    assert(result == nullptr);
    const char *msg = pscm_get_last_error_message();
    assert(msg != nullptr);
    printf("PASS: wrong arg count -> msg='%s'\n", msg);
  }

  // Test 4: valid code still returns non-NULL
  {
    SCM *result = pscm_eval_string("(+ 1 2)");
    assert(result != nullptr);
    assert((int64_t)result->value == 3);
    printf("PASS: valid code returns correct result\n");
  }

  // Test 5: quasiquote error returns NULL (missing argument)
  {
    SCM *result = pscm_eval_string("(quasiquote)");
    assert(result == nullptr);
    const char *msg = pscm_get_last_error_message();
    assert(msg != nullptr);
    printf("PASS: quasiquote error -> msg='%s'\n", msg);
  }

  // Test 6: pscm_parse error returns NULL
  {
    SCM *result = pscm_parse("(");
    assert(result == nullptr);
    const char *msg = pscm_get_last_error_message();
    assert(msg != nullptr);
    printf("PASS: pscm_parse error -> msg='%s'\n", msg);
  }

  printf("\nAll C API error tests passed!\n");
  return 0;
}
