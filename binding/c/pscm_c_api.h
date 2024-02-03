#pragma once

extern "C" {
void *pscm_create_scheme();
void pscm_destroy_scheme(void *scm);
void *pscm_eval(void *scm, const char *code);
const char *pscm_to_string(void *value);
}