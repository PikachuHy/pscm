#include "error.h"
#include "throw.h"
#include <stdarg.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// External globals written by the catch-all handler for public API boundary
extern char *g_last_error_message;
extern char *g_last_error_key;

static const int MAX_STACK_DEPTH = 100;
static int g_stack_depth = 0;

// Helper function to convert expression to string (with length limit)
static char *expr_to_string(SCM *expr, size_t max_len = 200) {
  if (!expr) {
    return nullptr;
  }

  // Use a temporary file to capture print_ast output
  FILE *tmp_file = tmpfile();
  if (!tmp_file) {
    return nullptr;
  }

  FILE *original_stdout = stdout;
  stdout = tmp_file;
  print_ast(expr, false);
  stdout = original_stdout;

  fseek(tmp_file, 0, SEEK_END);
  long file_size = ftell(tmp_file);
  fseek(tmp_file, 0, SEEK_SET);

  size_t buf_size = (file_size < (long)max_len) ? file_size + 1 : max_len + 5;
  char *buf = new char[buf_size];
  if (!buf) {
    fclose(tmp_file);
    return nullptr;
  }

  size_t read_size = fread(buf, 1, max_len, tmp_file);
  buf[read_size] = '\0';

  if (file_size > (long)max_len) {
    strcpy(buf + max_len, "...");
  }

  fclose(tmp_file);
  return buf;
}

// Push a frame onto the evaluation stack
void push_eval_stack(SCM *expr) {
  if (g_stack_depth >= MAX_STACK_DEPTH) {
    return;
  }

  if (!expr) {
    return;
  }

  // Only track expressions with source location to avoid saving pointers
  // to temporary stack-allocated expressions (which become invalid after return)
  const char *loc = get_source_location_str(expr);
  if (!loc) {
    return;
  }

  size_t loc_len = strlen(loc);
  char *loc_copy = new char[loc_len + 1];
  if (!loc_copy) {
    return;
  }
  strcpy(loc_copy, loc);

  char *expr_str = expr_to_string(expr, 200);

  EvalStackFrame *frame = new EvalStackFrame();
  if (!frame) {
    delete[] loc_copy;
    if (expr_str) delete[] expr_str;
    return;
  }

  frame->source_location = loc_copy;
  frame->expr_str = expr_str;
  frame->next = g_eval_stack;
  g_eval_stack = frame;
  g_stack_depth++;
}

// Pop a frame from the evaluation stack
void pop_eval_stack() {
  if (g_eval_stack) {
    EvalStackFrame *old = g_eval_stack;
    g_eval_stack = g_eval_stack->next;
    if (old->source_location) {
      delete[] old->source_location;
    }
    if (old->expr_str) {
      delete[] old->expr_str;
    }
    delete old;
    g_stack_depth--;
  }
}

// Print the evaluation call stack
void print_eval_stack() {
  if (!g_eval_stack) {
    fprintf(stderr, "\nEvaluation call stack: (empty)\n");
    return;
  }

  fprintf(stderr, "\nEvaluation call stack (most recent first):\n");
  EvalStackFrame *frame = g_eval_stack;
  int depth = 0;
  while (frame && depth < 20) {
    fprintf(stderr, "  #%d: ", depth);
    if (frame->source_location) {
      fprintf(stderr, "%s\n", frame->source_location);
    } else {
      fprintf(stderr, "<no source location>\n");
    }
    if (frame->expr_str) {
      fprintf(stderr, "      %s\n", frame->expr_str);
    }
    fprintf(stderr, "\n");
    frame = frame->next;
    depth++;
  }
  if (frame) {
    fprintf(stderr, "  ... (%d more frames)\n", g_stack_depth - depth);
  }
  fflush(stderr);
}

// Helper function to get type name as string
const char *get_type_name(SCM::Type type) {
  switch (type) {
    case SCM::NONE: return "none";
    case SCM::NIL: return "nil";
    case SCM::LIST: return "pair/list";
    case SCM::PROC: return "procedure";
    case SCM::CONT: return "continuation";
    case SCM::FUNC: return "function";
    case SCM::NUM: return "number";
    case SCM::FLOAT: return "float";
    case SCM::CHAR: return "character";
    case SCM::BOOL: return "boolean";
    case SCM::SYM: return "symbol";
    case SCM::STR: return "string";
    case SCM::MACRO: return "macro";
    case SCM::HASH_TABLE: return "hash-table";
    case SCM::RATIO: return "ratio";
    case SCM::VECTOR: return "vector";
    case SCM::PORT: return "port";
    case SCM::PROMISE: return "promise";
    case SCM::MODULE: return "module";
    default: return "unknown";
  }
}

void type_error(SCM *data, const char *expected_type) {
  char message[1024];
  int pos = 0;

  const char *loc_str = nullptr;
  if (data) {
    loc_str = get_source_location_str(data);
  }
  if (!loc_str && g_current_eval_context) {
    loc_str = get_source_location_str(g_current_eval_context);
  }

  if (loc_str) {
    pos += snprintf(message + pos, sizeof(message) - pos, "%s: ", loc_str);
  }

  pos += snprintf(message + pos, sizeof(message) - pos, "Type error: expected %s, but got ", expected_type);

  if (data) {
    const char *actual_type = get_type_name(data->type);
    pos += snprintf(message + pos, sizeof(message) - pos, "%s", actual_type);
  } else {
    pos += snprintf(message + pos, sizeof(message) - pos, "null");
  }

  if (g_current_eval_context && g_current_eval_context != data) {
    const char *ctx_loc = get_source_location_str(g_current_eval_context);
    if (ctx_loc) {
      pos += snprintf(message + pos, sizeof(message) - pos, "\n  While evaluating at %s", ctx_loc);
    }
  }

  eval_error("%s", message);
}

void eval_error(const char *format, ...) {
  va_list args;
  va_start(args, format);

  char message[1024];
  vsnprintf(message, sizeof(message), format, args);
  va_end(args);

  char full_message[2048];
  int pos = 0;

  if (g_current_eval_context) {
    const char *loc_str = get_source_location_str(g_current_eval_context);
    if (loc_str) {
      pos += snprintf(full_message + pos, sizeof(full_message) - pos, "%s: ", loc_str);
    }
  }

  pos += snprintf(full_message + pos, sizeof(full_message) - pos, "%s", message);

  SCM_String *s = (SCM_String *)gc_alloc(GC_STRING, sizeof(SCM_String));
  int len = (int)strlen(full_message);
  s->data = new char[len + 1];
  memcpy(s->data, full_message, len);
  s->data[len] = '\0';
  s->len = len;
  SCM *error_message = (SCM *)gc_alloc(GC_SCM, sizeof(SCM));
  error_message->type = SCM::STR;
  error_message->value = s;
  error_message->source_loc = nullptr;

  SCM_List *error_args = make_list(error_message);
  SCM *error_args_wrapped = wrap(error_args);

  if (g_error_key) {
    scm_throw(g_error_key, error_args_wrapped);
  }
  // Fallthrough: if scm_throw did not longjmp (should not happen normally
  // since g_error_key is set during init and the API boundary wraps in
  // catch-all), print the error and return so the host stays alive.
  fprintf(stderr, "%s\n", full_message);
  fprintf(stderr, "\n=== Evaluation Call Stack ===\n");
  if (g_eval_stack) {
    print_eval_stack();
  } else {
    fprintf(stderr, "Call stack is empty (error occurred at top level)\n");
  }
  fprintf(stderr, "=== End of Call Stack ===\n");
  fflush(stderr);
}

// Catch-all handler for public API boundary.
// Stores error details in globals so the host can retrieve them after
// pscm_eval/pscm_parse returns NULL.
SCM *scm_api_catch_handler(void *data, SCM *tag, SCM *args) {
  (void)data;

  // Free previous error info
  delete[] g_last_error_message;
  delete[] g_last_error_key;
  g_last_error_message = nullptr;
  g_last_error_key = nullptr;

  // Store error tag as a C string
  if (tag && is_sym(tag)) {
    const char *name = cast<SCM_Symbol>(tag)->data;
    g_last_error_key = new char[strlen(name) + 1];
    strcpy(g_last_error_key, name);
  }

  // Extract message from args (Guile convention: args is (msg ...))
  if (args && is_pair(args)) {
    SCM_List *args_list = cast<SCM_List>(args);
    if (args_list->data && is_str(args_list->data)) {
      SCM_String *s = cast<SCM_String>(args_list->data);
      g_last_error_message = new char[s->len + 1];
      memcpy(g_last_error_message, s->data, s->len);
      g_last_error_message[s->len] = '\0';
    }
  }

  return nullptr;
}
