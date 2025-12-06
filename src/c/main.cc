#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdnoreturn.h>
#include <string.h>

#include "pscm.h"

#include <assert.h>

bool debug_enabled = false;
bool ast_debug_enabled = false;
long *cont_base;

SCM_Environment g_env;

SCM *scm_none() {
  static SCM *value = nullptr;
  if (!value) {
    value = new SCM();
    value->type = SCM::NONE;
    value->value = nullptr;
  }
  return value;
}

SCM *scm_nil() {
  static SCM *value = nullptr;
  if (!value) {
    value = new SCM();
    value->type = SCM::NIL;
    value->value = nullptr;
  }
  return value;
}

SCM *scm_bool_false() {
  static SCM *value = nullptr;
  if (!value) {
    value = new SCM();
    value->type = SCM::BOOL;
    value->value = 0;
  }
  return value;
}

SCM *scm_bool_true() {
  static SCM *value = nullptr;
  if (!value) {
    value = new SCM();
    value->type = SCM::BOOL;
    value->value = (void *)1;
  }
  return value;
}

SCM_Symbol *make_sym(const char *data) {
  auto sym = new SCM_Symbol();
  size_t len = strlen(data);
  sym->data = new char[len + 1];
  memcpy(sym->data, data, len);
  sym->data[len] = '\0';  // Ensure null terminator
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
  return scm;
}

SCM *eval(SCM *ast) {
  auto env = &g_env;
  long stack_base;
  cont_base = &stack_base;
  auto ret = eval_with_env(env, ast);
  return ret;
}

// Helper function for type checking predicates
template <typename Predicate>
SCM *scm_c_type_check(SCM *arg, Predicate pred) {
  assert(arg);
  return pred(arg) ? scm_bool_true() : scm_bool_false();
}

SCM *scm_c_is_procedure(SCM *arg) {
  return scm_c_type_check(arg, [](SCM *a) { return is_proc(a) || is_cont(a) || is_func(a); });
}

SCM *scm_c_is_boolean(SCM *arg) {
  return scm_c_type_check(arg, is_bool);
}

SCM *scm_c_is_null(SCM *arg) {
  return scm_c_type_check(arg, is_nil);
}

SCM *scm_c_is_pair(SCM *arg) {
  return scm_c_type_check(arg, is_pair);
}

void init_scm() {
  g_env.parent = nullptr;
  g_env.dummy.data = nullptr;
  g_env.dummy.next = nullptr;
  scm_define_function("procedure?", 1, 0, 0, scm_c_is_procedure);
  scm_define_function("boolean?", 1, 0, 0, scm_c_is_boolean);
  scm_define_function("null?", 1, 0, 0, scm_c_is_null);
  scm_define_function("pair?", 1, 0, 0, scm_c_is_pair);
  scm_define_function("car", 1, 0, 0, car);
  scm_define_function("cdr", 1, 0, 0, cdr);
  scm_define_function("cadr", 1, 0, 0, cadr);
  init_number();
  init_eq();
  init_alist();
}

#define SCM_PRINT_AST(ast)                                                                                             \
  if (ast_debug_enabled) {                                                                                             \
    printf("[ast] ");                                                                                                  \
    print_ast(ast);                                                                                                    \
    printf("\n");                                                                                                      \
  }

void my_eval(SCM *ast) {
  SCM_PRINT_AST(ast);
  auto val = eval(ast);
  SCM_DEBUG_EVAL(" --> ");
  SCM_PRINT_AST(val);
  if (!is_none(val)) {
    print_ast(val);
    printf("\n");
  }
}

void repl() {
  char buffer[4096];  // Increase buffer size to avoid overflow
  while (true) {
    printf("pscm> ");
    if (fgets(buffer, sizeof(buffer), stdin) != NULL) {
      // Check if input is too long (fgets retains newline)
      size_t len = strlen(buffer);
      if (len > 0 && buffer[len - 1] != '\n') {
        // Input was truncated, clear remaining input
        int c;
        while ((c = getchar()) != '\n' && c != EOF) {
          // Discard remaining characters
        }
        printf("Warning: input too long, truncated\n");
      }
      auto ast = parse(buffer);
      print_ast(ast);
      printf("\n");
      auto val = eval(ast);
      printf(" --> ");
      print_ast(val);
      printf("\n");
    }
    else {
      // EOF or error, exit normally instead of calling exit()
      if (feof(stdin)) {
        printf("\n");
        return;  // Normal exit
      }
      printf("read failed\n");
      return;  // Return instead of calling exit()
    }
  }
}

void show_usage() {
  std::cout << R"(
Usage: pscm OPTION ...
Evaluate Scheme code, interactively or from a script.

  [-s] FILE      load Scheme source code from FILE, and exit

  --test FILE    load Scheme source code from FILE, print each eval result and exit
  --debug        enable debugging log output
  -h, --help     display this help and exit
  -v, --version  display version information and exit

please report bugs to https://github.com/PikachuHy/pscm/issues
)";
}

int do_eval(const char *filename) {
  init_scm();
  auto l = parse_file(filename);
  if (!l) {
    fprintf(stderr, "ERROR: failed to parse file: %s\n", filename);
    return 1;
  }
  while (l) {
    auto expr = l->data;
    my_eval(expr);
    assert(l);
    l = l->next;
  }
  return 0;
}

void setup_abort_handler();

int main(int argc, char **argv) {
  setup_abort_handler();
  if (argc < 2) {
    show_usage();
    return 1;  // Return error code when arguments are missing
  }

  int index = 1;
  char *filename = nullptr;
  while (index < argc) {
    std::string arg = argv[index];
    if (arg == "--debug") {
      debug_enabled = true;
      index++;
      continue;
    }
    else if (arg == "-h" || arg == "--help") {
      show_usage();
      return 0;
    }
    else if (arg == "-s" || arg == "--test") {
      if (index + 1 < argc) {
        if (filename) {
          fprintf(stderr, "ERROR: duplicate filename\n");
          return 1;
        }
        filename = argv[index + 1];
        index += 2;
      }
      else {
        fprintf(stderr, "ERROR: missing argument to `%s' switch\n", arg.c_str());
        show_usage();
        return 1;
      }
    }
    else {
      if (filename) {
        fprintf(stderr, "ERROR: duplicate filename\n");
        return 1;
      }
      filename = argv[index];
      index++;
    }
  }
  if (!filename) {
    fprintf(stderr, "ERROR: no filename specified\n");
    show_usage();
    return 1;
  }
  return do_eval(filename);
}
