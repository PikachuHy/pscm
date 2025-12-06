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
  return scm;
}

SCM *eval(SCM *ast) {
  long stack_base;
  cont_base = &stack_base;
  return eval_with_env(&g_env, ast);
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
  scm_define_function("cons", 2, 0, 0, scm_cons);
  scm_define_vararg_function("list", scm_list);
  scm_define_vararg_function("append", scm_append);
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
  printf("Usage: pscm OPTION ...\n");
  printf("Evaluate Scheme code, interactively or from a script.\n");
  printf("\n");
  printf("  [-s] FILE      load Scheme source code from FILE, and exit\n");
  printf("\n");
  printf("  --test FILE    load Scheme source code from FILE, print each eval result and exit\n");
  printf("  --debug        enable debugging log output\n");
  printf("  -h, --help     display this help and exit\n");
  printf("  -v, --version  display version information and exit\n");
  printf("\n");
  printf("please report bugs to https://github.com/PikachuHy/pscm/issues\n");
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
    const char *arg = argv[index];
    if (strcmp(arg, "--debug") == 0) {
      debug_enabled = true;
      index++;
      continue;
    }
    else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
      show_usage();
      return 0;
    }
    else if (strcmp(arg, "-s") == 0 || strcmp(arg, "--test") == 0) {
      if (index + 1 < argc) {
        if (filename) {
          fprintf(stderr, "ERROR: duplicate filename\n");
          return 1;
        }
        filename = argv[index + 1];
        index += 2;
      }
      else {
        fprintf(stderr, "ERROR: missing argument to `%s' switch\n", arg);
        show_usage();
        return 1;
      }
    }
    else {
      // Treat as filename
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
