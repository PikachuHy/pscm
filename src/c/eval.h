#pragma once

#include "pscm.h"

// do special form
SCM *eval_do(SCM_Environment *env, SCM_List *l);

// map special form
SCM *eval_map(SCM_Environment *env, SCM_List *l);

// Procedure and function application (used by map and other special forms)
SCM *apply_procedure(SCM_Environment *env, SCM_Procedure *proc, SCM_List *args);
SCM *eval_with_func(SCM_Function *func, SCM_List *l);

// Error handling (used by special forms)
[[noreturn]] void eval_error(const char *format, ...);

