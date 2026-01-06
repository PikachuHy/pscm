#pragma once

#include "pscm.h"

// Forward declarations
struct catch_info;

// Catch body function type
typedef SCM *(*scm_t_catch_body)(void *data);

// Catch handler function type
typedef SCM *(*scm_t_catch_handler)(void *data, SCM *tag, SCM *args);

// Core catch function (C API)
SCM *scm_c_catch(SCM *tag,
                 scm_t_catch_body body, void *body_data,
                 scm_t_catch_handler handler, void *handler_data);

// Throw exception
SCM *scm_throw(SCM *key, SCM *args);

// Uncaught throw handler (prints error and exits)
[[noreturn]] void scm_uncaught_throw(SCM *key, SCM *args);

// Internal catch functions (compatible with Guile 1.8 API)
SCM *scm_internal_catch(SCM *tag,
                       scm_t_catch_body body, void *body_data,
                       scm_t_catch_handler handler, void *handler_data);

SCM *scm_internal_lazy_catch(SCM *tag,
                            scm_t_catch_body body, void *body_data,
                            scm_t_catch_handler handler, void *handler_data);

// With throw handler (for lazy catch)
SCM *scm_c_with_throw_handler(SCM *tag,
                              scm_t_catch_body body, void *body_data,
                              scm_t_catch_handler handler, void *handler_data,
                              int lazy_catch_p);

// Scheme-callable catch function
SCM *scm_c_catch_scheme(SCM_List *args);

// Scheme-callable throw function
SCM *scm_c_throw_scheme(SCM_List *args);

// Initialize throw system
void init_throw();

// Standard error key (initialized in init_throw)
inline SCM *g_error_key = nullptr;

// Handle by message without exiting (unless quit tag)
SCM *scm_handle_by_message_noexit(void *handler_data, SCM *tag, SCM *args);

// Throw with noreturn flag (for lazy-catch compatibility)
SCM *scm_ithrow(SCM *key, SCM *args, int noreturn);

