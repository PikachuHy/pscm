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

// Call stack for tracking evaluation path
struct EvalStackFrame {
  char *source_location;        // Source location string (owned, must be freed)
  char *expr_str;               // String representation of expression (owned, must be freed)
  EvalStackFrame *next;         // Next frame in stack
};

// Print evaluation call stack (for debugging)
void print_eval_stack();

// External variables for error reporting context
extern SCM *g_current_eval_context;
extern EvalStackFrame *g_eval_stack;

// Helper function to print AST to stderr
inline void print_ast_to_stderr(SCM *ast) {
  // Temporarily redirect stdout to stderr for print_ast
  fflush(stdout);
  FILE *saved_stdout = stdout;
  stdout = stderr;
  print_ast(ast);
  fflush(stderr);
  stdout = saved_stdout;
}

// Macro expansion and definition
SCM *expand_macro_call(SCM_Environment *env, SCM_Macro *macro, SCM_List *args, SCM *original_call);
SCM *expand_macros(SCM_Environment *env, SCM *ast);
SCM *eval_define_macro(SCM_Environment *env, SCM_List *l);

// Define special form
SCM *eval_define(SCM_Environment *env, SCM_List *l);

// Lambda special form (used by define)
SCM *eval_lambda(SCM_Environment *env, SCM_List *l);

// delay special form (promises)
SCM *eval_delay(SCM_Environment *env, SCM_List *l);

// module special forms
SCM *eval_define_module(SCM_Environment *env, SCM_List *l);
SCM *eval_use_modules(SCM_Environment *env, SCM_List *l);
SCM *eval_export(SCM_Environment *env, SCM_List *l);
SCM *eval_define_public(SCM_Environment *env, SCM_List *l);

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
        // For non-symbol arguments, print the AST representation to stderr
        print_ast_to_stderr(current->data);
      }
    }
    else {
      fprintf(stderr, "()");
    }
    current = current->next;
  }
  fprintf(stderr, ")");
}

[[noreturn]] inline void report_arg_mismatch(SCM_List *expected, SCM_List *got, 
                                               const char *call_type = nullptr, 
                                               SCM *original_call = nullptr,
                                               SCM_Symbol *name = nullptr) {
  // Print source location if available
  const char *loc_str = nullptr;
  if (original_call) {
    loc_str = get_source_location_str(original_call);
  }
  if (!loc_str && g_current_eval_context) {
    loc_str = get_source_location_str(g_current_eval_context);
  }
  
  if (loc_str) {
    fprintf(stderr, "%s: ", loc_str);
  } else {
    fprintf(stderr, "<unknown location>: ");
  }
  
  // Print call type and name if available
  if (call_type) {
    fprintf(stderr, "%s argument mismatch", call_type);
    if (name) {
      fprintf(stderr, " for '%s'", name->data);
    }
    fprintf(stderr, "\n");
  } else {
    fprintf(stderr, "Argument mismatch\n");
  }
  
  // Print the original call expression if available
  // Check if original_call is a valid list/pair (not just a boolean or other value)
  bool printed_call = false;
  if (original_call && is_pair(original_call)) {
    fprintf(stderr, "  While calling: ");
    print_ast_to_stderr(original_call);
    fprintf(stderr, "\n");
    printed_call = true;
  }
  // Also print current eval context if different from original_call
  if (g_current_eval_context && is_pair(g_current_eval_context)) {
    if (!printed_call || g_current_eval_context != original_call) {
      fprintf(stderr, "  While evaluating: ");
      print_ast_to_stderr(g_current_eval_context);
      fprintf(stderr, "\n");
    }
  }
  
  fprintf(stderr, "  Expected arguments: ");
  print_arg_list(expected);
  fprintf(stderr, "\n");
  fprintf(stderr, "  But got arguments: ");
  print_arg_list(got);
  fprintf(stderr, "\n");
  
  // Print the evaluation call stack
  fprintf(stderr, "\n=== Evaluation Call Stack ===\n");
  if (g_eval_stack) {
    print_eval_stack();
  } else {
    fprintf(stderr, "Call stack is empty (error occurred at top level)\n");
  }
  fprintf(stderr, "=== End of Call Stack ===\n");
  fflush(stderr);
  
  exit(1);
}
