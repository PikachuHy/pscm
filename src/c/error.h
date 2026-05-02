#pragma once

#include "pscm.h"

// Call stack frame for tracking evaluation path
struct EvalStackFrame {
  char *source_location;        // Source location string (owned, must be freed)
  char *expr_str;               // String representation of expression (owned, must be freed)
  EvalStackFrame *next;         // Next frame in stack
};

// External variables for error reporting context
inline SCM *g_current_eval_context = nullptr;
inline EvalStackFrame *g_eval_stack = nullptr;

// Evaluation call stack management (used by eval_with_env)
void push_eval_stack(SCM *expr);
void pop_eval_stack();

// Print evaluation call stack (for debugging)
void print_eval_stack();

// Helper to get human-readable type name
const char *get_type_name(SCM::Type type);

// Error handling functions
[[noreturn]] void eval_error(const char *format, ...);
[[noreturn]] void type_error(SCM *data, const char *expected_type);

// Helper function to print AST to stderr
inline void print_ast_to_stderr(SCM *ast) {
  fflush(stdout);
  FILE *saved_stdout = stdout;
  stdout = stderr;
  print_ast(ast);
  fflush(stderr);
  stdout = saved_stdout;
}
