#include "pscm.h"
#include "eval.h"

// GC root helpers from other translation units
void register_module_roots();
void register_port_roots();
void register_env_roots();

void init_scm() {
  gc_init();

  g_env.parent = nullptr;
  g_env.dummy.data = nullptr;
  g_env.dummy.next = nullptr;
  init_predicate();
  init_list();
  init_symbol();
  init_apply();
  init_map();
  init_number();
  init_continuation();
  init_throw();  // Initialize catch/throw exception handling
  init_eq();
  init_alist();
  init_char();
  init_string();
  init_port();
  init_delay();
  init_eval();
  init_macro();  // Initialize macro functions (macroexpand-1, macroexpand)
  init_values();
  init_hash_table();
  init_procedure();
  init_vector();
  init_modules();  // Initialize modules before load, so %load-path can be defined in module
  init_load();     // Initialize load after modules, so %load-path can be defined in current module
  init_smob();     // Initialize smob system
  init_variable(); // Initialize variable system (scm_variable_ref, scm_c_lookup)
  init_read_options(); // Initialize read options system (read-set!, read-enable, read-disable)
  init_debug_options(); // Initialize debug options system (debug-set!, debug-enable, debug-disable)
  init_exit();

  // Register (gc) built-in to trigger garbage collection from Scheme.
  scm_define_function("gc", 0, 0, 0, scm_gc);

  // Register explicit GC roots after all builtins are registered
  gc_register_root(&g_root_module, "g_root_module");
  gc_register_root((SCM **)&g_wind_chain, "g_wind_chain");
  gc_register_root(&g_current_eval_context, "g_current_eval_context");
  extern SCM *g_error_key;
  gc_register_root(&g_error_key, "g_error_key");

  // Register roots from other translation units
  register_module_roots();
  register_port_roots();
  register_env_roots();
}

