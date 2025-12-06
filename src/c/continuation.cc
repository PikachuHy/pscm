#include "pscm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern long *cont_base;

long scm_stack_size(long *start) {
  long stack;
  SCM_DEBUG_CONT("stack_top: %p\n", &stack);
  return start - &stack;
}

SCM *scm_make_continuation(int *first) {
  long stack_size = scm_stack_size(cont_base);
  long *src;
  auto cont = make_cont(stack_size, (long *)malloc(sizeof(long) * stack_size));
  auto data = wrap(cont);
  SCM_DEBUG_CONT("stack_len: %ld\n", stack_size);
  src = cont_base;
  src -= stack_size;
  memcpy((void *)cont->stack_data, src, sizeof(long) * cont->stack_len);
  *first = !setjmp(cont->cont_jump_buffer);
  if (*first) {
    return data;
  }
  else {
    auto ret = cont->arg;
    return ret;
  }
}

void grow_stack(SCM *cont, SCM *args);

void copy_stack_and_call(SCM_Continuation *cont, SCM *args, long *dst) {
  cont->arg = args;
  memcpy(dst, cont->stack_data, sizeof(long) * cont->stack_len);
  long __cur__;
  SCM_DEBUG_CONT("jump from %p to %p use %p with %lu\n", &__cur__, dst, cont, cont->stack_len);
  longjmp(cont->cont_jump_buffer, 1);
}

void scm_dynthrow(SCM *cont, SCM *args) {
  auto continuation = cast<SCM_Continuation>(cont);
  long *dst = cont_base;
  long stack_top_element;
  dst -= continuation->stack_len;
  if (dst <= &stack_top_element) {
    grow_stack(cont, args);
  }
  copy_stack_and_call(continuation, args, dst);
}

void grow_stack(SCM *cont, SCM *args) {
  long growth[100];
  scm_dynthrow(cont, args);
}
