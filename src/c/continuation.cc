#include "pscm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern long *cont_base;
extern SCM_List *g_wind_chain;

long scm_stack_size(long *start) {
  long stack;
  SCM_DEBUG_CONT("stack_top: %p\n", &stack);
  return start - &stack;
}

// Forward declaration
SCM_List *copy_wind_chain(SCM_List *chain);
SCM_List *unwind_wind_chain(SCM_List *target);
void rewind_wind_chain(SCM_List *common, SCM_List *target);

SCM *scm_make_continuation(int *first) {
  long stack_size = scm_stack_size(cont_base);
  long *src;
  auto cont = make_cont(stack_size, (long *)malloc(sizeof(long) * stack_size));
  auto data = wrap(cont);
  SCM_DEBUG_CONT("stack_len: %ld\n", stack_size);
  src = cont_base;
  src -= stack_size;
  memcpy((void *)cont->stack_data, src, sizeof(long) * cont->stack_len);
  
  // Save current wind chain
  cont->wind_chain = copy_wind_chain(g_wind_chain);
  
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
  
  // Handle wind chain: unwind current to common prefix, then rewind to continuation's wind chain
  SCM_List *common = unwind_wind_chain(continuation->wind_chain);
  rewind_wind_chain(common, continuation->wind_chain);
  
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
