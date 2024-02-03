#include "pscm_c_api.h"
#include <pscm/Scheme.h>
using namespace pscm;

void *pscm_create_scheme() {
  return new Scheme();
}

void pscm_destroy_scheme(void *scm) {
  auto p = (Scheme *)scm;
  delete p;
}

void *pscm_eval(void *scm, const char *code) {
  auto p = (Scheme *)scm;
  auto ret = p->eval(code);
  return new Cell(ret);
}

const char *pscm_to_string(void *value) {
  auto p = (Cell *)value;
  auto s = p->to_std_string();
  char *c_str = new char[s.size() + 1];
  strcpy(c_str, s.c_str());
  c_str[s.size()] = '\0';
  return c_str;
}