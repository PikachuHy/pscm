#include "pscm.h"
#include "eval.h"

// Common function to evaluate a parsed expression list
// Returns the result of the last expression evaluated, or scm_none() if no expressions
SCM *scm_eval_expression_list(SCM_List *expr_list) {
  if (!expr_list) {
    return scm_none();
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

// primitive-load: core file loading implementation (accepts const char *)
SCM *scm_c_primitive_load(const char *filename) {
  // Debug log: print the file path being loaded
  fprintf(stderr, "[load] Loading file: %s\n", filename);
  
  // Use existing parse_file helper to get list of expressions
  SCM_List *expr_list = parse_file(filename);
  if (!expr_list) {
    eval_error("primitive-load: failed to load file: %s", filename);
  }

  return scm_eval_expression_list(expr_list);
}

// Helper function: load from SCM string (used by Scheme-level load/primitive-load)
SCM *scm_primitive_load(SCM *filename) {
  if (!is_str(filename)) {
    eval_error("primitive-load: expected string");
  }

  SCM_String *s = cast<SCM_String>(filename);
  const char *fname = s->data;
  
  return scm_c_primitive_load(fname);
}

// Initialize file loading functions
void init_load() {
  // primitive-load: low-level file loading, used by load
  scm_define_function("primitive-load", 1, 0, 0, scm_primitive_load);
  // load: user-facing interface, initial implementation is same as primitive-load
  scm_define_function("load", 1, 0, 0, scm_primitive_load);
  
  // Define %load-path variable (initialized to empty list, like Guile 1.8)
  // Always define in root module (pscm-user) so it's accessible from all modules
  // Get root module directly (g_root_module is set in init_modules)
  if (!g_root_module || !is_module(g_root_module)) {
    eval_error("init_load: root module not initialized");
    return;
  }
  SCM_Module *root_module = cast<SCM_Module>(g_root_module);
  
  // Define in root module's obarray
  SCM_Symbol *load_path_sym = make_sym("%load-path");
  scm_c_hash_set_eq(wrap(root_module->obarray), wrap(load_path_sym), scm_nil());
}



