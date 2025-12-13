#include "pscm.h"
#include "eval.h"
#include <stdlib.h>

// exit: Exit the program with a status code
SCM *scm_c_exit(SCM_List *args) {
  int status = 0;
  
  // exit can take 0 or 1 argument (the exit status)
  if (args && args->data) {
    if (is_num(args->data)) {
      status = (int)(intptr_t)args->data->value;
    } else if (is_float(args->data)) {
      status = (int)ptr_to_double(args->data->value);
    } else {
      eval_error("exit: expected integer");
      return nullptr;
    }
  }
  
  // Exit the program
  exit(status);
  return nullptr;  // Never reached
}

void init_exit() {
  scm_define_vararg_function("exit", scm_c_exit);
}

