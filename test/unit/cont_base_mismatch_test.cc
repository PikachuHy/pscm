/*
 * Test: continuation invocation across scm_c_eval_string calls from
 * different C stack depths.
 *
 * Bug: cont_base is a global set by each eval to a stack-local variable.
 * When capture and invocation eval calls are at different C depths,
 * scm_dynthrow writes the saved stack to the wrong position.
 *
 * Reproduction: even 1 extra C frame triggers the segfault.
 */

#include "pscm_api.h"
#include "pscm.h"
#include <stdio.h>
#include <stdlib.h>

SCM *eval(SCM *ast) {
  return pscm_eval(ast);
}

static int failures = 0;

#define CHECK(cond, msg)                               \
  do {                                                 \
    if (!(cond)) {                                     \
      fprintf(stderr, "FAIL: %s\n", msg);              \
      failures++;                                      \
    } else {                                           \
      printf("PASS: %s\n", msg);                       \
    }                                                  \
  } while (0)

static void capture_from_c_depth(int depth) {
  if (depth == 0) {
    scm_c_eval_string(
        "(define k2 #f)"
        "(call/cc (lambda (c) (set! k2 c) 'captured-from-c-depth))");
  } else {
    capture_from_c_depth(depth - 1);
  }
}

int main(void) {
  /* 'phase' must be above cont_base so the stack copy doesn't restore it.
     This is the same mechanism that lets do_eval's loop variable advance
     correctly across continuation invocations. */
  volatile int phase = 0;
  long main_stack_ref;
  cont_base = &main_stack_ref;
  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stderr, NULL, _IONBF, 0);

  pscm_init();

  /*
   * Capture wrapped in 1 extra C frame, invoke from direct context.
   * Different C depth between capture and invocation: the capture
   * scm_c_eval_string is called from capture_from_c_depth(1) while
   * the invocation is called directly from main.
   */
  capture_from_c_depth(1);

  if (phase == 0) {
    phase = 1;
    scm_c_eval_string("(k2 'invoked)");
    // never reached — continuation returns through capture_from_c_depth
  } else {
    printf("  cross-depth continuation returned (no crash)\n");
  }

  printf("Results: %d failures\n", failures);
  return failures > 0 ? 1 : 0;
}
