#include "pscm.h"
#include "eval.h"
#include <ctype.h>

SCM *scm_c_is_char(SCM *arg) {
  return is_char(arg) ? scm_bool_true() : scm_bool_false();
}

SCM *scm_c_char_to_integer(SCM *arg) {
  if (!is_char(arg)) {
    eval_error("char->integer: expected character");
    return nullptr;
  }
  char ch = scm_to_char(arg);
  SCM *scm = new SCM();
  scm->type = SCM::NUM;
  scm->value = (void*)(int64_t)(unsigned char)ch;
  scm->source_loc = nullptr;
  return scm;
}

SCM *scm_c_integer_to_char(SCM *arg) {
  if (!is_num(arg)) {
    eval_error("integer->char: expected integer");
    return nullptr;
  }
  int64_t val = (int64_t)arg->value;
  if (val < 0 || val > 255) {
    eval_error("integer->char: value out of range [0, 255]");
    return nullptr;
  }
  return scm_from_char((char)val);
}

SCM *scm_c_char_upcase(SCM *arg) {
  if (!is_char(arg)) {
    eval_error("char-upcase: expected character");
    return nullptr;
  }
  char ch = scm_to_char(arg);
  char upper = (char)toupper((unsigned char)ch);
  return scm_from_char(upper);
}

SCM *scm_c_char_downcase(SCM *arg) {
  if (!is_char(arg)) {
    eval_error("char-downcase: expected character");
    return nullptr;
  }
  char ch = scm_to_char(arg);
  char lower = (char)tolower((unsigned char)ch);
  return scm_from_char(lower);
}

void init_char() {
  scm_define_function("char?", 1, 0, 0, scm_c_is_char);
  scm_define_function("char->integer", 1, 0, 0, scm_c_char_to_integer);
  scm_define_function("integer->char", 1, 0, 0, scm_c_integer_to_char);
  scm_define_function("char-upcase", 1, 0, 0, scm_c_char_upcase);
  scm_define_function("char-downcase", 1, 0, 0, scm_c_char_downcase);
}

