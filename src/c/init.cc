#include "pscm.h"
#include "eval.h"

extern SCM_Environment g_env;

void init_scm() {
  g_env.parent = nullptr;
  g_env.dummy.data = nullptr;
  g_env.dummy.next = nullptr;
  init_predicate();
  init_list();
  init_symbol();
  init_apply();
  init_number();
  init_eq();
  init_alist();
  init_char();
  init_string();
  init_eval();
  init_hash_table();
}

