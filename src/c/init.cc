#include "pscm.h"
#include "eval.h"


void init_scm() {
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
  init_exit();
}

