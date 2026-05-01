#include "pscm.h"
#include "eval.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


long scm_stack_size(long *start) {
  long stack;
  SCM_DEBUG_CONT("stack_top: %p\n", &stack);
  return start - &stack;
}

// Forward declaration
SCM_List *copy_wind_chain(SCM_List *chain);
SCM_List *unwind_wind_chain(SCM_List *target);
void rewind_wind_chain(SCM_List *common, SCM_List *target);

// no_sanitize("address"): this function reads the entire C stack (from
// cont_base to the current stack top), crossing multiple independent stack
// frames. This is UB per the C standard and ASan detects it as
// stack-buffer-underflow. We suppress ASan here and use a manual byte loop
// instead of memcpy because ASan intercepts memcpy at link time and
// function-level no_sanitize cannot suppress the interceptor.
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
__attribute__((no_sanitize("address")))
#endif
#endif
SCM *scm_make_continuation(int *first) {
  long stack_size = scm_stack_size(cont_base);
  long *src;
  SCM_DEBUG_CONT("stack_len: %ld\n", stack_size);
  src = cont_base;
  src -= stack_size;
  auto cont = make_cont(stack_size, (long *)malloc(sizeof(long) * stack_size), src);
  auto data = wrap(cont);
#if defined(__has_feature) && __has_feature(address_sanitizer)
  {
    volatile char *dst_bytes = (volatile char *)cont->stack_data;
    volatile char *src_bytes = (volatile char *)src;
    size_t n = sizeof(long) * cont->stack_len;
    for (size_t i = 0; i < n; i++) {
      dst_bytes[i] = src_bytes[i];
    }
  }
#else
  memcpy(cont->stack_data, src, sizeof(long) * cont->stack_len);
#endif
  
  // Save current wind chain
  cont->wind_chain = copy_wind_chain(g_wind_chain);
  
  // Save current module (needed for variable lookup after continuation restore)
  cont->saved_module = scm_current_module();
  
  *first = !setjmp(cont->cont_jump_buffer);
  if (*first) {
    return data;
  }
  else {
    // Restore current module when continuation is invoked
    if (cont->saved_module) {
      scm_set_current_module(cont->saved_module);
    }
    auto ret = cont->arg;
    return ret;
  }
}

void grow_stack(SCM *cont, SCM *args);

// no_sanitize("address"): counterpart to scm_make_continuation — writes
// saved stack data back to the C stack. Manual byte copy to avoid ASan's
// memcpy interceptor.
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
__attribute__((no_sanitize("address")))
#endif
#endif
void copy_stack_and_call(SCM_Continuation *cont, SCM *args, long *dst) {
  cont->arg = args;
#if defined(__has_feature) && __has_feature(address_sanitizer)
  {
    volatile char *dst_bytes = (volatile char *)dst;
    volatile char *src_bytes = (volatile char *)cont->stack_data;
    size_t n = sizeof(long) * cont->stack_len;
    for (size_t i = 0; i < n; i++) {
      dst_bytes[i] = src_bytes[i];
    }
  }
#else
  memcpy(dst, cont->stack_data, sizeof(long) * cont->stack_len);
#endif
  long __cur__;
  SCM_DEBUG_CONT("jump from %p to %p use %p with %lu\n", &__cur__, dst, cont, cont->stack_len);
  longjmp(cont->cont_jump_buffer, 1);
}

void scm_dynthrow(SCM *cont, SCM *args) {
  auto continuation = cast<SCM_Continuation>(cont);

  // Handle wind chain: unwind current to common prefix, then rewind to continuation's wind chain
  SCM_List *common = unwind_wind_chain(continuation->wind_chain);
  rewind_wind_chain(common, continuation->wind_chain);

  long *dst = continuation->stack_src;
  long stack_top_element;
  if (dst <= &stack_top_element) {
    grow_stack(cont, args);
  }
  copy_stack_and_call(continuation, args, dst);
}

// no_sanitize("address"): forces stack growth by allocating a stack array
// so there is room to restore the continuation's saved stack. Suppress
// ASan to prevent it from treating the growth array as a stack overflow.
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
__attribute__((no_sanitize("address")))
#endif
#endif
void grow_stack(SCM *cont, SCM *args) {
  long growth[100];
  (void)growth; // prevent compiler from optimizing away the array
  scm_dynthrow(cont, args);
}

// Wrapper function for call-with-current-continuation as a builtin function
// This function is called when call-with-current-continuation is used as a value (not as a special form)
SCM *scm_c_call_with_current_continuation(SCM_List *args) {
  // This function should never be called directly
  // It's registered as a placeholder, and eval.cc will handle it specially
  eval_error("call-with-current-continuation: internal error - should be handled as special form");
  return nullptr;
}

void init_continuation() {
  scm_define_vararg_function("call-with-current-continuation", scm_c_call_with_current_continuation);
  scm_define_vararg_function("call/cc", scm_c_call_with_current_continuation);
}
