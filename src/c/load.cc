#include "pscm.h"
#include "eval.h"

extern SCM_Environment g_env;

// primitive-load: core file loading implementation used by Scheme-level load
SCM *scm_c_primitive_load(SCM *filename) {
  if (!is_str(filename)) {
    eval_error("primitive-load: expected string");
  }

  SCM_String *s = cast<SCM_String>(filename);
  const char *fname = s->data;

  // Use existing parse_file helper to get list of expressions
  SCM_List *expr_list = parse_file(fname);
  if (!expr_list) {
    eval_error("primitive-load: failed to load file: %s", fname);
  }

  // Evaluate each expression in the top-level environment
  SCM *result = scm_none();
  SCM_Environment *top_env = g_env.parent ? g_env.parent : &g_env;
  SCM_List *it = expr_list;
  while (it) {
    result = eval_with_env(top_env, it->data);
    it = it->next;
  }

  return result;
}

// Initialize file loading functions
void init_load() {
  // primitive-load: low-level file loading, used by load
  scm_define_function("primitive-load", 1, 0, 0, scm_c_primitive_load);
  // load: user-facing interface, initial implementation is same as primitive-load
  scm_define_function("load", 1, 0, 0, scm_c_primitive_load);
}



