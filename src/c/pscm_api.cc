#include "pscm_api.h"
#include "error.h"
#include "eval.h"
#include "throw.h"
#include <stdio.h>
#include <stdlib.h>

// Global state definitions are now in pscm.h as inline variables

// Error handler callback
static pscm_error_handler_t g_error_handler = nullptr;

// Last error storage (for retrieval after pscm_eval returns NULL).
// Non-static — the catch-all handler in error.cc writes to these.
char *g_last_error_message = nullptr;
char *g_last_error_key = nullptr;

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
  if (!cont_base) {
    long stack_base;
    cont_base = &stack_base;
  }
  return scm_c_catch(
      scm_bool_true(),
      [](void *data) -> SCM * {
        return eval_with_env(&g_env, (SCM *)data);
      },
      (void *)ast,
      scm_api_catch_handler,
      nullptr);
}

// Parse and evaluate a string
SCM *pscm_eval_string(const char *code) {
  if (!code) {
    return nullptr;
  }
  return scm_c_catch(
      scm_bool_true(),
      [](void *data) -> SCM * {
        const char *c = (const char *)data;
        SCM *ast = pscm_parse(c);
        if (!ast) {
          return nullptr;
        }
        return pscm_eval(ast);
      },
      (void *)code,
      scm_api_catch_handler,
      nullptr);
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
  return scm_c_catch(
      scm_bool_true(),
      [](void *data) -> SCM * {
        return parse((const char *)data);
      },
      (void *)code,
      scm_api_catch_handler,
      nullptr);
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

SCM *pscm_c_define(const char *name, SCM *val) {
  return scm_c_define(name, val);
}

SCM *pscm_c_define_gsubr(const char *name, int req, int opt, int rst,
                         SCM *(*fcn)(void)) {
  return scm_c_define_gsubr(name, req, opt, rst, fcn);
}

// Module operations (compatible with Guile 1.8 API)
SCM *pscm_c_resolve_module(const char *name) {
  return scm_c_resolve_module(name);
}

SCM *pscm_c_module_lookup(SCM *module, const char *name) {
  return scm_c_module_lookup(module, name);
}

SCM *pscm_c_module_define(SCM *module, const char *name, SCM *val) {
  return scm_c_module_define(module, name, val);
}

void pscm_c_use_module(const char *name) {
  scm_c_use_module(name);
}

// Port operations (compatible with Guile 1.8 API)
SCM *pscm_current_error_port(void) {
  return scm_current_error_port();
}

SCM *pscm_set_current_error_port(SCM *port) {
  return scm_set_current_error_port(port);
}

SCM *pscm_current_input_port(void) {
  return scm_current_input_port();
}
SCM *pscm_set_current_input_port(SCM *port) {
  return scm_set_current_input_port(port);
}
SCM *pscm_current_output_port(void) {
  return scm_current_output_port();
}
SCM *pscm_set_current_output_port(SCM *port) {
  return scm_set_current_output_port(port);
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

const char *pscm_get_last_error_key(void) {
  return g_last_error_key;
}

const char *pscm_get_last_error_message(void) {
  return g_last_error_message;
}

