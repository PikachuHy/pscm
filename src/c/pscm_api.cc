#include "pscm_api.h"
#include "eval.h"
#include "throw.h"
#include <stdio.h>
#include <stdlib.h>

// Global state - these are defined in main.cc when building the executable
// For library use, they should be defined elsewhere or made thread-local
extern bool debug_enabled;
extern bool ast_debug_enabled;
extern long *cont_base;
extern SCM_Environment g_env;
extern SCM_List *g_wind_chain;

// Error handler callback
static pscm_error_handler_t g_error_handler = nullptr;

// Library initialization
void pscm_init(void) {
  init_scm();
}

// Library cleanup (currently no-op, but reserved for future use)
void pscm_cleanup(void) {
  // Future: cleanup resources, close files, etc.
}

// Evaluate an AST node
SCM *pscm_eval(SCM *ast) {
  if (!ast) {
    return nullptr;
  }
  // Set stack base for continuation support
  long stack_base;
  cont_base = &stack_base;
  return eval_with_env(&g_env, ast);
}

// Parse and evaluate a string
SCM *pscm_eval_string(const char *code) {
  if (!code) {
    return nullptr;
  }
  SCM *ast = pscm_parse(code);
  if (!ast) {
    return nullptr;
  }
  return pscm_eval(ast);
}

// Parse and evaluate a file
SCM *pscm_eval_file(const char *filename) {
  if (!filename) {
    return nullptr;
  }
  SCM_List *exprs = pscm_parse_file(filename);
  if (!exprs) {
    return nullptr;
  }
  
  // Evaluate all expressions, return the last result
  SCM *last_result = nullptr;
  SCM_List *current = exprs;
  while (current) {
    if (current->data) {
      last_result = pscm_eval(current->data);
    }
    current = current->next;
  }
  return last_result;
}

// Parse a string into an AST
SCM *pscm_parse(const char *code) {
  if (!code) {
    return nullptr;
  }
  return parse(code);
}

// Parse a file into a list of AST nodes
SCM_List *pscm_parse_file(const char *filename) {
  if (!filename) {
    return nullptr;
  }
  return parse_file(filename);
}

// Get the global environment
SCM_Environment *pscm_get_global_env(void) {
  return &g_env;
}

// Create a new environment with a parent
SCM_Environment *pscm_create_env(SCM_Environment *parent) {
  if (!parent) {
    parent = &g_env;
  }
  return make_env(parent);
}

// Set error handler
void pscm_set_error_handler(pscm_error_handler_t handler) {
  g_error_handler = handler;
}

// Debugging control
void pscm_set_debug_enabled(bool enabled) {
  debug_enabled = enabled;
}

void pscm_set_ast_debug_enabled(bool enabled) {
  ast_debug_enabled = enabled;
}

bool pscm_get_debug_enabled(void) {
  return debug_enabled;
}

bool pscm_get_ast_debug_enabled(void) {
  return ast_debug_enabled;
}

// Variable operations (compatible with Guile 1.8 API)
SCM *pscm_c_lookup(const char *name) {
  return scm_c_lookup(name);
}

SCM *pscm_variable_ref(SCM *var) {
  return scm_variable_ref(var);
}

SCM *pscm_make_variable(SCM *init) {
  return scm_make_variable(init);
}

SCM *pscm_make_undefined_variable(void) {
  return scm_make_undefined_variable();
}

// Port operations (compatible with Guile 1.8 API)
SCM *pscm_current_error_port(void) {
  return scm_current_error_port();
}

SCM *pscm_set_current_error_port(SCM *port) {
  return scm_set_current_error_port(port);
}

SCM *pscm_force_output(SCM *port) {
  // Create a list wrapper for the port argument
  SCM_List args;
  args.data = port;
  args.next = nullptr;
  args.is_dotted = false;
  return scm_force_output(&args);
}

// Throw operations (compatible with Guile 1.8 API)
SCM *pscm_ithrow(SCM *key, SCM *args, int noreturn) {
  return scm_ithrow(key, args, noreturn);
}

