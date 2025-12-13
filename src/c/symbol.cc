#include "pscm.h"
#include "eval.h"
#include <string.h>
#include <stdio.h>

// Helper macro to define singleton SCM values
#define DEFINE_SCM_SINGLETON(func_name, type_val, value_val) \
  SCM *scm_##func_name() { \
    static SCM *value = nullptr; \
    if (!value) { \
      value = new SCM(); \
      value->type = SCM::type_val; \
      value->value = value_val; \
    } \
    return value; \
  }

DEFINE_SCM_SINGLETON(none, NONE, nullptr)
DEFINE_SCM_SINGLETON(nil, NIL, nullptr)
DEFINE_SCM_SINGLETON(bool_false, BOOL, 0)
DEFINE_SCM_SINGLETON(bool_true, BOOL, (void *)1)

SCM_Symbol *make_sym(const char *data) {
  auto sym = new SCM_Symbol();
  int len = (int)strlen(data);
  sym->data = new char[len + 1];
  strcpy(sym->data, data);
  sym->len = len;
  return sym;
}

// Helper macro to define symbol singleton functions
#define DEFINE_SYMBOL_SINGLETON(func_name, symbol_name) \
  SCM *scm_sym_##func_name() { \
    static SCM *value = nullptr; \
    if (!value) { \
      value = wrap(make_sym(symbol_name)); \
    } \
    return value; \
  }

DEFINE_SYMBOL_SINGLETON(let, "let")
DEFINE_SYMBOL_SINGLETON(quote, "quote")
DEFINE_SYMBOL_SINGLETON(quasiquote, "quasiquote")
DEFINE_SYMBOL_SINGLETON(unquote, "unquote")
DEFINE_SYMBOL_SINGLETON(unquote_splicing, "unquote-splicing")
DEFINE_SYMBOL_SINGLETON(letrec, "letrec")
DEFINE_SYMBOL_SINGLETON(set, "set!")
DEFINE_SYMBOL_SINGLETON(lambda, "lambda")

SCM *create_sym(const char *data, int len) {
  SCM_Symbol *sym = new SCM_Symbol();
  sym->data = new char[len + 1];
  memcpy(sym->data, data, len);
  sym->data[len] = '\0';  // Ensure null terminator
  sym->len = len;
  SCM *scm = new SCM();
  scm->type = SCM::SYM;
  scm->value = sym;
  scm->source_loc = nullptr;  // Initialize to nullptr
  return scm;
}

SCM *scm_c_gensym() {
  static int index = 0;
  char buffer[32];
  snprintf(buffer, sizeof(buffer), " g%d", index++);
  return wrap(make_sym(buffer));
}

// Helper function to create a string from C string
static SCM *scm_from_c_string(const char *data, int len) {
  SCM_String *s = new SCM_String();
  s->data = new char[len + 1];
  memcpy(s->data, data, len);
  s->data[len] = '\0';
  s->len = len;
  SCM *scm = new SCM();
  scm->type = SCM::STR;
  scm->value = s;
  scm->source_loc = nullptr;
  return scm;
}

// symbol->string: Convert symbol to string
SCM *scm_c_symbol_to_string(SCM *sym) {
  if (!is_sym(sym)) {
    eval_error("symbol->string: expected symbol");
    return nullptr;
  }
  auto s = cast<SCM_Symbol>(sym);
  return scm_from_c_string(s->data, s->len);
}

// string->symbol: Convert string to symbol
SCM *scm_c_string_to_symbol(SCM *str) {
  if (!is_str(str)) {
    eval_error("string->symbol: expected string");
    return nullptr;
  }
  auto s = cast<SCM_String>(str);
  return create_sym(s->data, s->len);
}

void init_symbol() {
  scm_define_function("gensym", 0, 0, 0, scm_c_gensym);
  scm_define_function("symbol->string", 1, 0, 0, scm_c_symbol_to_string);
  scm_define_function("string->symbol", 1, 0, 0, scm_c_string_to_symbol);
}

