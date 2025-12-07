#include "pscm.h"
#include "eval.h"

// Helper function to apply procedure with arguments
SCM *apply_procedure(SCM_Environment *env, SCM_Procedure *proc, SCM_List *args) {
  auto proc_env = make_env(proc->env);
  auto args_l = proc->args;
  // Save original args for error reporting - create a copy of the list structure
  // (we only need the structure, not the values, for error reporting)
  SCM_List *original_args = args; // Save original args pointer for error reporting
  SCM_List *args_iter = args;     // Use separate iterator for the loop
  while (args_iter && args_l) {
    assert(is_sym(args_l->data));
    auto arg_sym = cast<SCM_Symbol>(args_l->data);
    auto arg_val = eval_with_env(env, args_iter->data);
    scm_env_insert(proc_env, arg_sym, arg_val, /*search_parent=*/false);
    if (debug_enabled) {
      SCM_DEBUG_EVAL("bind func arg ");
      printf("%s to ", arg_sym->data);
      print_ast(arg_val);
      printf("\n");
    }
    args_iter = args_iter->next;
    args_l = args_l->next;
  }
  if (args_iter || args_l) {
    report_arg_mismatch(proc->args, original_args);
  }
  return eval_with_list(proc_env, proc->body);
}

