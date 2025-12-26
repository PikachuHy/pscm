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

// Error handling
typedef void (*pscm_error_handler_t)(const char *message);
void pscm_set_error_handler(pscm_error_handler_t handler);

// Debugging
void pscm_set_debug_enabled(bool enabled);
void pscm_set_ast_debug_enabled(bool enabled);
bool pscm_get_debug_enabled(void);
bool pscm_get_ast_debug_enabled(void);

#ifdef __cplusplus
}
#endif

#endif // PSCM_API_H

