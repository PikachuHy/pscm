#pragma once

#include "pscm.h"

// do special form
SCM *eval_do(SCM_Environment *env, SCM_List *l);

// map special form
SCM *eval_map(SCM_Environment *env, SCM_List *l);

// for-each special form
SCM *eval_for_each(SCM_Environment *env, SCM_List *l);

// apply special form
SCM *eval_apply(SCM_Environment *env, SCM_List *l);

// call-with-values special form
SCM *eval_call_with_values(SCM_Environment *env, SCM_List *l);

// dynamic-wind special form
SCM *eval_dynamic_wind(SCM_Environment *env, SCM_List *l);

// cond special form
SCM *eval_cond(SCM_Environment *env, SCM_List *l, SCM **ast);

// case special form
SCM *eval_case(SCM_Environment *env, SCM_List *l);

// Procedure and function application (used by map and other special forms)
SCM *apply_procedure(SCM_Environment *env, SCM_Procedure *proc, SCM_List *args);
SCM *eval_with_func(SCM_Function *func, SCM_List *l);

// Error handling (used by special forms)
[[noreturn]] void eval_error(const char *format, ...);
[[noreturn]] void type_error(SCM *data, const char *expected_type);

// Macro expansion and definition
SCM *expand_macro_call(SCM_Environment *env, SCM_Macro *macro, SCM_List *args, SCM *original_call);
SCM *expand_macros(SCM_Environment *env, SCM *ast);
SCM *eval_define_macro(SCM_Environment *env, SCM_List *l);

// Define special form
SCM *eval_define(SCM_Environment *env, SCM_List *l);

// Lambda special form (used by define)
SCM *eval_lambda(SCM_Environment *env, SCM_List *l);

// List evaluation (used by macros and other special forms)
SCM *eval_with_list(SCM_Environment *env, SCM_List *l);

// Helper functions for argument mismatch reporting (used by apply_procedure and macros)
inline void print_arg_list(SCM_List *l) {
  if (!l) {
    fprintf(stderr, "()");
    return;
  }
  fprintf(stderr, "(");
  bool first = true;
  SCM_List *current = l; // Use a separate variable to avoid modifying the original pointer
  while (current) {
    if (!first) {
      fprintf(stderr, " ");
    }
    first = false;
    if (current->data) {
      if (is_sym(current->data)) {
        SCM_Symbol *sym = cast<SCM_Symbol>(current->data);
        fprintf(stderr, "%s", sym->data);
      }
      else {
        // For non-symbol arguments, print the AST representation
        print_ast(current->data);
      }
    }
    else {
      fprintf(stderr, "()");
    }
    current = current->next;
  }
  fprintf(stderr, ")");
}

[[noreturn]] inline void report_arg_mismatch(SCM_List *expected, SCM_List *got) {
  fprintf(stderr, "args not match\n");
  fprintf(stderr, "expect ");
  print_arg_list(expected);
  fprintf(stderr, "\n");
  fprintf(stderr, "but got ");
  print_arg_list(got);
  fprintf(stderr, "\n");
  exit(1);
}
