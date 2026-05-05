#ifndef PSCM_API_H
#define PSCM_API_H

#include "pscm.h"

#ifdef __cplusplus
extern "C" {
#endif

// Library initialization and cleanup
void pscm_init(void);
void pscm_cleanup(void);

// Evaluation functions
SCM *pscm_eval(SCM *ast);
SCM *pscm_eval_string(const char *code);
SCM *pscm_eval_file(const char *filename);

// Parsing functions
SCM *pscm_parse(const char *code);
SCM_List *pscm_parse_file(const char *filename);

// Environment access
SCM_Environment *pscm_get_global_env(void);
SCM_Environment *pscm_create_env(SCM_Environment *parent);

// Variable operations (compatible with Guile 1.8 API)
SCM *pscm_c_lookup(const char *name);  // Look up a variable by name, returns variable object
SCM *pscm_variable_ref(SCM *var);      // Get the value of a variable object
SCM *pscm_make_variable(SCM *init);    // Create a variable initialized to a value
SCM *pscm_make_undefined_variable(void); // Create an unbound variable

// Define a variable from C (Guile 1.8 compatible)
SCM *pscm_c_define(const char *name, SCM *val);

// Register a C function as a Scheme procedure (Guile 1.8 compatible)
SCM *pscm_c_define_gsubr(const char *name, int req, int opt, int rst,
                         SCM *(*fcn)(void));

// Module operations (compatible with Guile 1.8 API)
SCM *pscm_c_resolve_module(const char *name);                         // Resolve module by name string
SCM *pscm_c_module_lookup(SCM *module, const char *name);             // Look up variable in module

// Port operations (compatible with Guile 1.8 API)
SCM *pscm_current_error_port(void);    // Get current error port
SCM *pscm_set_current_error_port(SCM *port); // Set current error port
SCM *pscm_current_input_port(void);    // Get current input port
SCM *pscm_set_current_input_port(SCM *port); // Set current input port
SCM *pscm_current_output_port(void);   // Get current output port
SCM *pscm_set_current_output_port(SCM *port); // Set current output port
SCM *pscm_force_output(SCM *port);     // Force output (flush) a port

// Throw operations (compatible with Guile 1.8 API)
SCM *pscm_ithrow(SCM *key, SCM *args, int noreturn); // Throw with noreturn flag

// Error handling
typedef void (*pscm_error_handler_t)(const char *message);
void pscm_set_error_handler(pscm_error_handler_t handler);

// Error retrieval (after pscm_eval/pscm_parse returns NULL)
const char *pscm_get_last_error_key(void);
const char *pscm_get_last_error_message(void);

// Debugging
void pscm_set_debug_enabled(bool enabled);
void pscm_set_ast_debug_enabled(bool enabled);
bool pscm_get_debug_enabled(void);
bool pscm_get_ast_debug_enabled(void);

#ifdef __cplusplus
}
#endif

#endif // PSCM_API_H

