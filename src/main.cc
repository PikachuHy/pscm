#include "pscm_api.h"

// Global state (defined here for the executable)
bool debug_enabled = false;
bool ast_debug_enabled = false;
long *cont_base = nullptr;
SCM_Environment g_env;
SCM_List *g_wind_chain = nullptr;

// Wrapper for backward compatibility (used by repl.cc)
SCM *eval(SCM *ast) {
  return pscm_eval(ast);
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
    // In test mode, use write format (with quotes for strings)
    // This matches the expected output format in tests
    print_ast(val, true);  // true = write mode (with quotes for strings)
    printf("\n");
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
  pscm_init();
  auto l = pscm_parse_file(filename);
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
    // No arguments: enter REPL mode
    pscm_init();
    repl();
    return 0;
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
