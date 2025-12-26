#include "pscm.h"

#include "eval.h"
#include <stdarg.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

// Helper function to print AST to stderr (moved to eval.h as inline function)

SCM *eval_with_list(SCM_Environment *env, SCM_List *l) {
  assert(l);
  SCM *ret = nullptr;
  while (l) {
    ret = eval_with_env(env, l->data);
    l = l->next;
  }
  return ret;
}

SCM_List *eval_list_with_env(SCM_Environment *env, SCM_List *l) {
  SCM_List dummy = make_list_dummy();
  SCM_List *it = &dummy;
  while (l) {
    auto val = eval_with_env(env, l->data);
    auto next = make_list(val);
    it->next = next;
    it = it->next;
    l = l->next;
  }
  return dummy.next;
}


// Error handling helper with context
SCM *g_current_eval_context = nullptr;

// Call stack for tracking evaluation path (EvalStackFrame defined in eval.h)
EvalStackFrame *g_eval_stack = nullptr;
static const int MAX_STACK_DEPTH = 100;  // Prevent infinite recursion
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
  
  // Save original stdout
  FILE *original_stdout = stdout;
  
  // Redirect stdout to temporary file
  stdout = tmp_file;
  
  // Print expression
  print_ast(expr, false);
  
  // Restore stdout
  stdout = original_stdout;
  
  // Get file size
  fseek(tmp_file, 0, SEEK_END);
  long file_size = ftell(tmp_file);
  fseek(tmp_file, 0, SEEK_SET);
  
  // Allocate buffer (add 1 for null terminator, and 4 for "...")
  size_t buf_size = (file_size < (long)max_len) ? file_size + 1 : max_len + 5;
  char *buf = new char[buf_size];
  if (!buf) {
    fclose(tmp_file);
    return nullptr;
  }
  
  // Read from temporary file
  size_t read_size = fread(buf, 1, max_len, tmp_file);
  buf[read_size] = '\0';
  
  // If truncated, add "..."
  if (file_size > (long)max_len) {
    strcpy(buf + max_len, "...");
  }
  
  fclose(tmp_file);
  return buf;
}

// Push a frame onto the evaluation stack
static void push_eval_stack(SCM *expr) {
  if (g_stack_depth >= MAX_STACK_DEPTH) {
    return;  // Stack too deep, don't add more frames
  }
  
  if (!expr) {
    return;  // Skip null expressions
  }
  
  // Only track expressions with source location to avoid saving pointers
  // to temporary stack-allocated expressions (which become invalid after return)
  // The problem: expr might be a stack-allocated temporary (like in hash_table.cc:359)
  // where `SCM call_expr;` is a local variable on the stack.
  // 
  // Solution: Try to get source location, but if it fails or crashes,
  // we'll skip tracking this expression. We use a conservative approach:
  // only track if we can safely get the source location.
  //
  // Note: get_source_location_str checks for null and source_loc existence,
  // so it should be safe. But if expr is corrupted, even this could crash.
  // In practice, if expr is valid enough to be passed to eval_with_env,
  // it should be safe to check source_loc.
  const char *loc = get_source_location_str(expr);
  if (!loc) {
    // Skip expressions without source location (usually temporary expressions)
    // This avoids saving pointers to stack-allocated expressions that become invalid
    return;
  }
  
  // Immediately copy the location string before any other function calls
  // that might overwrite the static buffer in get_source_location_str
  size_t loc_len = strlen(loc);
  char *loc_copy = new char[loc_len + 1];
  if (!loc_copy) {
    return;  // Memory allocation failed
  }
  strcpy(loc_copy, loc);
  
  // Convert expression to string (with length limit)
  char *expr_str = expr_to_string(expr, 200);
  
  EvalStackFrame *frame = new EvalStackFrame();
  if (!frame) {
    delete[] loc_copy;
    if (expr_str) delete[] expr_str;
    return;
  }
  
  frame->source_location = loc_copy;
  frame->expr_str = expr_str;  // Can be nullptr if conversion failed
  frame->next = g_eval_stack;
  g_eval_stack = frame;
  g_stack_depth++;
}

// Pop a frame from the evaluation stack
static void pop_eval_stack() {
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
  while (frame && depth < 20) {  // Limit to 20 frames for readability
    fprintf(stderr, "  #%d: ", depth);
    if (frame->source_location) {
      fprintf(stderr, "%s\n", frame->source_location);
    } else {
      fprintf(stderr, "<no source location>\n");
    }
    // Print expression if available
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
static const char *get_type_name(SCM::Type type) {
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

[[noreturn]] void type_error(SCM *data, const char *expected_type) {
  // Try to get source location from the data itself first
  const char *loc_str = nullptr;
  if (data) {
    loc_str = get_source_location_str(data);
  }
  
  // If not available, try current eval context
  if (!loc_str && g_current_eval_context) {
    loc_str = get_source_location_str(g_current_eval_context);
  }
  
  if (loc_str) {
    fprintf(stderr, "%s: ", loc_str);
  } else {
    fprintf(stderr, "<unknown location>: ");
  }
  
  fprintf(stderr, "Type error: expected %s, but got ", expected_type);
  
  if (data) {
    const char *actual_type = get_type_name(data->type);
    fprintf(stderr, "%s", actual_type);
    fprintf(stderr, "\n  Value: ");
    print_ast(data);
    fprintf(stderr, "\n");
  } else {
    fprintf(stderr, "null\n");
  }
  
  // Also print current eval context if available and different
  if (g_current_eval_context && g_current_eval_context != data) {
    const char *ctx_loc = get_source_location_str(g_current_eval_context);
    if (ctx_loc) {
      fprintf(stderr, "  While evaluating at %s: ", ctx_loc);
    } else {
      fprintf(stderr, "  While evaluating: ");
    }
    print_ast(g_current_eval_context);
    fprintf(stderr, "\n");
  }
  
  exit(1);
}

[[noreturn]] void eval_error(const char *format, ...) {
  va_list args;
  va_start(args, format);

  // Print source location if available
  bool printed_location = false;
  if (g_current_eval_context) {
    const char *loc_str = get_source_location_str(g_current_eval_context);
    if (loc_str) {
      fprintf(stderr, "%s: ", loc_str);
      printed_location = true;
    }
    fprintf(stderr, "Error while evaluating: ");
    print_ast_to_stderr(g_current_eval_context);
    fprintf(stderr, "\n");
    if (!printed_location) {
      fprintf(stderr, "  (no source location available)\n");
    }
    fprintf(stderr, "  ");
  }
  else {
    fprintf(stderr, "%s:%d: ", __FILE__, __LINE__);
  }

  vfprintf(stderr, format, args);
  fprintf(stderr, "\n");
  va_end(args);
  
  // Print the evaluation call stack
  fprintf(stderr, "\n=== Evaluation Call Stack ===\n");
  if (g_eval_stack) {
    print_eval_stack();
  } else {
    fprintf(stderr, "Call stack is empty (error occurred at top level)\n");
  }
  fprintf(stderr, "=== End of Call Stack ===\n");
  fflush(stderr);
  
  exit(1);
}

// Helper functions for special forms
static SCM *eval_quote(SCM_List *l) {
  return l->next ? l->next->data : scm_nil();
}

SCM *eval_quasiquote(SCM_Environment *env, SCM_List *l);

static SCM *eval_set(SCM_Environment *env, SCM_List *l) {
  assert(l->next && is_sym(l->next->data));
  auto sym = cast<SCM_Symbol>(l->next->data);
  auto val = eval_with_env(env, l->next->next->data);
  
  // First try to update in environment (searches parent environments)
  auto entry = scm_env_search_entry(env, sym, /*search_parent=*/true);
  if (entry) {
    entry->value = val;
  } else {
    // Not found in environment, check if it's in the current module
    SCM *current_mod = scm_current_module();
    if (current_mod && is_module(current_mod)) {
      SCM_Module *module = cast<SCM_Module>(current_mod);
      SCM_Module *var_module = module_find_variable_module(module, sym);
      if (var_module) {
        // Variable exists in a module, update it
        scm_c_hash_set_eq(wrap(var_module->obarray), wrap(sym), val);
      } else {
        // Variable not found in any module, create new binding in environment
        scm_env_insert(env, sym, val, /*search_parent=*/false);
      }
    } else {
      // No module, create new binding in environment
      scm_env_insert(env, sym, val, /*search_parent=*/false);
    }
  }
  
  if (debug_enabled) {
    SCM_DEBUG_EVAL("set! ");
    printf("%s to ", sym->data);
    print_ast(val);
    printf("\n");
  }
  // set! returns an unspecified value, which should not be printed
  return scm_none();
}

SCM *eval_lambda(SCM_Environment *env, SCM_List *l) {
  SCM *param_spec = l->next->data;
  SCM_List *proc_sig = nullptr;
  
  // Check if parameter is a single symbol (e.g., (lambda x body))
  // In Scheme, (lambda x body) is equivalent to (lambda (. x) body)
  if (is_sym(param_spec)) {
    // Convert single symbol to a rest parameter list: (. symbol)
    auto rest_sym = cast<SCM_Symbol>(param_spec);
    SCM_List *rest_param_node = make_list(param_spec);
    rest_param_node->is_dotted = true;  // Mark as rest parameter
    proc_sig = rest_param_node;
  } else if (is_pair(param_spec) || is_nil(param_spec)) {
    // Normal parameter list (list or nil)
    proc_sig = cast<SCM_List>(param_spec);
  } else {
    type_error(param_spec, "symbol, pair, or nil");
  }
  
  auto proc = make_proc(nullptr, proc_sig, l->next->next, env);
  auto ret = wrap(proc);
  if (debug_enabled) {
    SCM_DEBUG_EVAL("create proc ");
    print_ast(ret);
    printf(" from ");
    print_list(l);
    printf("\n");
  }
  return ret;
}


// Helper function to count list length
// Helper function to lookup symbol in environment
static SCM *lookup_symbol(SCM_Environment *env, SCM_Symbol *sym) {
  auto val = scm_env_search(env, sym);
  if (!val) {
    eval_error("symbol '%s' not found", sym->data);
  }
  return val;
}

// Helper function for if special form
static SCM *eval_if(SCM_Environment *env, SCM_List *l, SCM **ast) {
  assert(l->next);
  auto pred = eval_with_env(env, l->next->data);
  // In Scheme, only #f is falsy, everything else is truthy
  if (is_truthy(pred)) {
    *ast = l->next->next->data;
    return nullptr; // Signal to continue evaluation
  }
  if (l->next->next->next) {
    *ast = l->next->next->next->data;
    return nullptr; // Signal to continue evaluation
  }
  return scm_none();
}

// Helper function for and special form
static SCM *eval_and(SCM_Environment *env, SCM_List *l, SCM **ast) {
  if (!l->next) {
    // No arguments, return #t
    return scm_bool_true();
  }
  SCM_List *current = l->next;
  // Evaluate all expressions except the last one
  while (current && current->next) {
    SCM *result = eval_with_env(env, current->data);
    // If any expression evaluates to #f (falsy), return #f immediately
    // In Scheme, only #f is falsy, everything else is truthy
    if (is_falsy(result)) {
      return scm_bool_false();
    }
    current = current->next;
  }
  // Evaluate and return the last expression
  if (current) {
    *ast = current->data;
    return nullptr; // Signal to continue evaluation
  }
  return scm_bool_true();
}

// Helper function for or special form
static SCM *eval_or(SCM_Environment *env, SCM_List *l, SCM **ast) {
  if (!l->next) {
    // No arguments, return #f
    return scm_bool_false();
  }
  SCM_List *current = l->next;
  // Evaluate expressions until we find one that's truthy
  while (current && current->next) {
    SCM *result = eval_with_env(env, current->data);
    // If any expression evaluates to truthy (not #f), return it immediately
    // In Scheme, only #f is falsy, everything else is truthy
    if (is_truthy(result)) {
      return result;
    }
    current = current->next;
  }
  // Evaluate and return the last expression
  if (current) {
    *ast = current->data;
    return nullptr; // Signal to continue evaluation
  }
  return scm_bool_false();
}

// Helper function for call/cc special form
static SCM *eval_call_cc(SCM_Environment *env, SCM_List *l, SCM **ast) {
  assert(l->next);
  auto proc = eval_with_env(env, l->next->data);
  int first;
  auto cont = scm_make_continuation(&first);
  SCM_DEBUG_CONT("jump back: ");
  if (!first) {
    if (debug_enabled) {
      printf("cont is ");
      print_ast(cont);
      printf("\n");
    }
    return cont;
  }
  *ast = scm_list2(proc, cont);
  if (debug_enabled) {
    print_ast(*ast);
    printf("\n");
  }
  return nullptr; // Signal to continue evaluation
}


SCM *eval_with_env(SCM_Environment *env, SCM *ast) {
  assert(env);
  assert(ast);

  // Push onto evaluation stack for call trace
  push_eval_stack(ast);

  // Save current context for error reporting
  SCM *old_context = g_current_eval_context;
  g_current_eval_context = ast;

// Helper macro to restore context before returning
#define RETURN_WITH_CONTEXT(val)                                                                                       \
  do {                                                                                                                 \
    g_current_eval_context = old_context;                                                                              \
    pop_eval_stack();                                                                                                  \
    return (val);                                                                                                      \
  } while (0)

entry:
  SCM_DEBUG_EVAL("eval ");
  if (debug_enabled) {
    print_ast(ast);
    printf("\n");
  }
  if (!is_pair(ast)) {
    if (is_sym(ast)) {
      SCM_Symbol *sym = cast<SCM_Symbol>(ast);
      // Keywords (symbols starting with ':') are self-evaluating
      if (sym->data && sym->data[0] == ':') {
        RETURN_WITH_CONTEXT(ast);
      }
      SCM *result = lookup_symbol(env, sym);
      RETURN_WITH_CONTEXT(result);
    }
    RETURN_WITH_CONTEXT(ast);
  }
  SCM_List *l = cast<SCM_List>(ast);
  assert(l->data);
  if (is_sym(l->data)) {
    SCM_Symbol *sym = cast<SCM_Symbol>(l->data);

    // Check if this is a macro call (before checking special forms)
    SCM *val = scm_env_exist(env, sym);
    if (val && is_macro(val)) {
      // Found a macro, expand it and continue evaluation
      SCM_Macro *macro = cast<SCM_Macro>(val);
      SCM *expanded = expand_macro_call(env, macro, l->next, ast);
      // Recursively expand the result, then evaluate
      ast = expand_macros(env, expanded);
      // If macro expansion returns a non-pair value (e.g., a number), 
      // we need to handle it directly instead of trying to evaluate it as a list
      if (!is_pair(ast)) {
        RETURN_WITH_CONTEXT(ast);
      }
      // After macro expansion, we need to re-check the structure
      // If ast is a pair but l->data is not a symbol/proc/func, it might be
      // a self-evaluating value wrapped in a list, which should be evaluated directly
      // We need to recast l since ast might have changed
      l = cast<SCM_List>(ast);
      if (l && l->data && !is_sym(l->data) && !is_proc(l->data) && !is_func(l->data) && !is_cont(l->data) && !is_pair(l->data) && !l->next) {
        // This is a self-evaluating value (number, string, etc.) wrapped in a single-element list
        // Return it directly
        RETURN_WITH_CONTEXT(l->data);
      }
      goto entry;
    }

    if (is_sym_val(l->data, "define")) {
      SCM *result = eval_define(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "define-macro")) {
      SCM *result = eval_define_macro(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "define-module")) {
      SCM *result = eval_define_module(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "use-modules")) {
      SCM *result = eval_use_modules(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "export")) {
      SCM *result = eval_export(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "re-export")) {
      SCM *result = eval_re_export(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "define-public")) {
      SCM *result = eval_define_public(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "let")) {
      ast = expand_let(ast);
      goto entry;
    }
    else if (is_sym_val(l->data, "let*")) {
      ast = expand_letstar(ast);
      goto entry;
    }
    else if (is_sym_val(l->data, "letrec")) {
      ast = expand_letrec(ast);
      goto entry;
    }
    else if (is_sym_val(l->data, "call/cc") || is_sym_val(l->data, "call-with-current-continuation")) {
      auto ret = eval_call_cc(env, l, &ast);
      if (ret) {
        RETURN_WITH_CONTEXT(ret);
      }
      goto entry;
    }
    else if (is_sym_val(l->data, "lambda")) {
      SCM *result = eval_lambda(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "set!")) {
      SCM *result = eval_set(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "quote")) {
      SCM *result = eval_quote(l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "quasiquote")) {
      SCM *result = eval_quasiquote(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "if")) {
      auto ret = eval_if(env, l, &ast);
      if (ret) {
        RETURN_WITH_CONTEXT(ret);
      }
      goto entry;
    }
    else if (is_sym_val(l->data, "and")) {
      auto ret = eval_and(env, l, &ast);
      if (ret) {
        RETURN_WITH_CONTEXT(ret);
      }
      goto entry;
    }
    else if (is_sym_val(l->data, "or")) {
      auto ret = eval_or(env, l, &ast);
      if (ret) {
        RETURN_WITH_CONTEXT(ret);
      }
      goto entry;
    }
    else if (is_sym_val(l->data, "delay")) {
      SCM *result = eval_delay(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "cond")) {
      auto ret = eval_cond(env, l, &ast);
      if (ret) {
        RETURN_WITH_CONTEXT(ret);
      }
      goto entry;
    }
    else if (is_sym_val(l->data, "case")) {
      SCM *result = eval_case(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "begin")) {
      // begin: evaluate all expressions in sequence, return the last one
      if (!l->next) {
        // No expressions, return #f
        RETURN_WITH_CONTEXT(scm_bool_false());
      }
      SCM_List *current = l->next;
      SCM *result = scm_bool_false();
      // Evaluate all expressions except the last one
      while (current && current->next) {
        eval_with_env(env, current->data);
        current = current->next;
      }
      // Evaluate and return the last expression
      if (current) {
        result = eval_with_env(env, current->data);
      }
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "for-each")) {
      SCM *result = eval_for_each(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "do")) {
      SCM *result = eval_do(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "map")) {
      SCM *result = eval_map(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "map-in-order")) {
      SCM *result = eval_map(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "apply")) {
      SCM *result = eval_apply(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "call-with-values")) {
      SCM *result = eval_call_with_values(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else if (is_sym_val(l->data, "dynamic-wind")) {
      SCM *result = eval_dynamic_wind(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    else {
      // Variable reference: resolve symbol and build call expression
      // (val was already looked up above for macro check)
      if (!val) {
        val = lookup_symbol(env, sym);
      }
      
      // Safety check: validate val before using it
      if (!val) {
        eval_error("symbol '%s' not found", sym->data);
        return nullptr;
      }
      
      // Check if val has a valid type before creating new list
      // Use try-catch to safely check type
      bool type_valid = false;
      try {
        if (val->type >= SCM::NONE && val->type <= SCM::MODULE) {
          type_valid = true;
        }
      } catch (...) {
        type_valid = false;
      }
      
      if (!type_valid) {
        fprintf(stderr, "Error: symbol '%s' resolved to invalid value with type %d (0x%x) at %p\n",
                sym->data, 
                (val ? (int)val->type : -1), 
                (val ? (unsigned int)val->type : 0), 
                (void *)val);
        fprintf(stderr, "  This suggests the symbol lookup returned a corrupted pointer.\n");
        fprintf(stderr, "  Valid type range: %d (NONE) to %d (MODULE)\n", 
                (int)SCM::NONE, (int)SCM::MODULE);
        fflush(stderr);
        eval_error("symbol '%s' resolved to corrupted value", sym->data);
        return nullptr;
      }
      
      auto new_list = make_list(val);
      new_list->next = l->next;
      ast = wrap(new_list);
      goto entry;
    }
  }
  else if (is_cont(l->data)) {
    SCM *cont_arg = scm_nil();
    if (l->next) {
      cont_arg = eval_with_env(env, l->next->data);
    }
    scm_dynthrow(l->data, cont_arg);
    return nullptr;  // Never reached, but satisfies compiler
  }
  else if (is_proc(l->data)) {
    auto proc = cast<SCM_Procedure>(l->data);
    SCM *result = apply_procedure(env, proc, l->next);
    RETURN_WITH_CONTEXT(result);
  }
  else if (is_func(l->data)) {
    auto func = cast<SCM_Function>(l->data);
    // Special handling for apply: it needs access to environment and handles argument evaluation itself
    if (func->name && strcmp(func->name->data, "apply") == 0) {
      // For apply, when called as a function value, arguments are already evaluated
      // We need to pass them to eval_apply, which will handle them correctly
      // l->data is the apply function, l->next contains the already-evaluated arguments
      SCM *result = eval_apply(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    if (func->name && strcmp(func->name->data, "map") == 0) {
      // For map, when called as a function value, arguments are already evaluated
      // We need to pass them to eval_map, which will handle them correctly
      // l->data is the map function, l->next contains the already-evaluated arguments
      SCM *result = eval_map(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    if (func->name && strcmp(func->name->data, "map-in-order") == 0) {
      // For map-in-order, when called as a function value, arguments are already evaluated
      // We need to pass them to eval_map, which will handle them correctly
      // l->data is the map-in-order function, l->next contains the already-evaluated arguments
      SCM *result = eval_map(env, l);
      RETURN_WITH_CONTEXT(result);
    }
    if (func->name && (strcmp(func->name->data, "call-with-current-continuation") == 0 || 
                       strcmp(func->name->data, "call/cc") == 0)) {
      // For call-with-current-continuation, when called as a function value, arguments are already evaluated
      // We need to handle it specially, similar to how it's handled as a special form
      // l->data is the call-with-current-continuation function, l->next contains the already-evaluated arguments
      if (!l->next) {
        eval_error("call-with-current-continuation: requires 1 argument (procedure)");
        RETURN_WITH_CONTEXT(nullptr);
      }
      // The argument is already evaluated, so we can use it directly
      // But we need to create a continuation and call the procedure with it
      SCM *proc = l->next->data;
      if (!is_proc(proc) && !is_func(proc)) {
        eval_error("call-with-current-continuation: argument must be a procedure");
        RETURN_WITH_CONTEXT(nullptr);
      }
      // Create continuation and call procedure
      int first;
      auto cont = scm_make_continuation(&first);
      if (!first) {
        // Continuation was invoked, return the argument
        RETURN_WITH_CONTEXT(cont);
      }
      // Call procedure with continuation
      SCM_List *cont_list = make_list(cont);
      SCM_List proc_call;
      proc_call.data = proc;
      proc_call.next = cont_list;
      if (is_proc(proc)) {
        SCM *result = apply_procedure(env, cast<SCM_Procedure>(proc), cont_list);
        RETURN_WITH_CONTEXT(result);
      } else {
        // For function, evaluate arguments and call
        SCM_List *evaled_args = eval_list_with_env(env, cont_list);
        SCM_List func_call;
        func_call.data = proc;
        func_call.next = evaled_args;
        SCM *result = eval_with_func(cast<SCM_Function>(proc), &func_call);
        RETURN_WITH_CONTEXT(result);
      }
    }
    auto func_argl = eval_list_with_env(env, l->next);
    if (debug_enabled) {
      SCM_DEBUG_EVAL(" ");
      printf("before eval args: ");
      print_list(l->next);
      printf("\n");
      printf("after eval args: ");
      print_list(func_argl);
      printf("\n");
    }
    l->next = func_argl;
    SCM *result = eval_with_func(func, l);
    RETURN_WITH_CONTEXT(result);
  }
  else if (is_pair(l->data)) {
    // Nested list: evaluate first element and continue
    auto f = eval_with_env(env, l->data);
    auto new_l = make_list(f);
    new_l->next = l->next;
    ast = wrap(new_l);
    goto entry;
  }
  else if (is_num(l->data) || is_float(l->data) || is_str(l->data) || is_char(l->data) || is_bool(l->data) || is_nil(l->data)) {
    // Self-evaluating values: numbers, strings, characters, booleans, nil
    // If this is a single-element list with a self-evaluating value, return it directly
    if (!l->next) {
      RETURN_WITH_CONTEXT(l->data);
    }
    // Otherwise, this is an error - can't have a list starting with a self-evaluating value
    // Fall through to error reporting
  }
  else {
    // Enhanced error reporting for unsupported expression types
    SCM *expr = l ? l->data : nullptr;
    const char *type_name = "unknown";
    const char *loc_str = nullptr;
    const char *list_loc_str = nullptr;
    
    // First, immediately print basic error info to stderr and flush
    // This ensures we see something even if accessing expr crashes
    fprintf(stderr, "<unknown location>: Error: not supported expression type: ");
    fflush(stderr);
    
    // Safely get type name - check if expr is valid before accessing
    if (expr) {
      // Check if expr is a valid pointer by checking if it's within reasonable bounds
      // This is a basic sanity check - in practice, if expr is corrupted, we might still crash
      // but at least we try to get the type safely
      try {
        type_name = get_type_name(expr->type);
        fprintf(stderr, "%s\n", type_name);
      } catch (...) {
        type_name = "corrupted";
        fprintf(stderr, "%s\n", type_name);
      }
    } else {
      type_name = get_type_name(SCM::NONE);
      fprintf(stderr, "%s\n", type_name);
    }
    fflush(stderr);
    
    // Now try to get location info - but if it crashes, we've at least printed the error
    // Add safety check: validate expr->type before accessing expr->source_loc
    if (expr) {
      // First check if type is valid before accessing source_loc
      // This prevents crashes from accessing corrupted memory
      bool type_valid = false;
      try {
        // Check if type is within valid enum range
        if (expr->type >= SCM::NONE && expr->type <= SCM::MODULE) {
          type_valid = true;
        }
      } catch (...) {
        // If accessing expr->type crashes, skip location info
        type_valid = false;
      }
      
      if (type_valid) {
        try {
          loc_str = get_source_location_str(expr);
        } catch (...) {
          // If accessing source_loc crashes, just skip location info
          loc_str = nullptr;
        }
      }
    }
    
    // Try to get source location from the list structure
    if (l) {
      try {
        SCM *list_wrapped = wrap(l);
        if (list_wrapped) {
          // Check if list_wrapped type is valid before accessing source_loc
          bool list_type_valid = false;
          try {
            if (list_wrapped->type >= SCM::NONE && list_wrapped->type <= SCM::MODULE) {
              list_type_valid = true;
            }
          } catch (...) {
            list_type_valid = false;
          }
          
          if (list_type_valid) {
            try {
              list_loc_str = get_source_location_str(list_wrapped);
            } catch (...) {
              list_loc_str = nullptr;
            }
          }
        }
      } catch (...) {
        // If wrap or accessing crashes, just skip
        list_loc_str = nullptr;
      }
    }
    
    // If still no location, try current eval context
    if (!loc_str && !list_loc_str && g_current_eval_context) {
      try {
        // Check if context type is valid before accessing source_loc
        bool ctx_type_valid = false;
        try {
          if (g_current_eval_context->type >= SCM::NONE && g_current_eval_context->type <= SCM::MODULE) {
            ctx_type_valid = true;
          }
        } catch (...) {
          ctx_type_valid = false;
        }
        
        if (ctx_type_valid) {
          try {
            loc_str = get_source_location_str(g_current_eval_context);
          } catch (...) {
            loc_str = nullptr;
          }
        }
      } catch (...) {
        // If accessing context crashes, just skip
        loc_str = nullptr;
      }
    }
    
    // If we got location info, print it (but we already printed error above)
    if (loc_str || list_loc_str) {
      fprintf(stderr, "  Location: %s\n", loc_str ? loc_str : list_loc_str);
    }
    
    // Print the problematic expression with its location
    // Use try-catch to handle potential crashes from corrupted pointers
    if (expr) {
      fprintf(stderr, "\n  Problematic expression");
      if (loc_str) {
        fprintf(stderr, " (at %s)", loc_str);
      }
      fprintf(stderr, ": ");
      try {
        print_ast_to_stderr(expr);
      } catch (...) {
        fprintf(stderr, "<unable to print - corrupted pointer>");
      }
    } else {
      fprintf(stderr, "\n  Expression is null");
    }
    
    // Print the full list structure for context with its location
    fprintf(stderr, "\n  Full expression");
    if (list_loc_str) {
      fprintf(stderr, " (at %s)", list_loc_str);
    }
    fprintf(stderr, ": ");
    try {
      if (l) {
        print_ast_to_stderr(wrap(l));
      } else {
        fprintf(stderr, "<list is null>");
      }
    } catch (...) {
      fprintf(stderr, "<unable to print - corrupted pointer>");
    }
    
    // Print the full expression being evaluated (if different from expr)
    if (g_current_eval_context) {
      try {
        SCM *ctx_wrapped = l ? wrap(l) : nullptr;
        if (g_current_eval_context != expr && g_current_eval_context != ctx_wrapped) {
          const char *ctx_loc = nullptr;
          try {
            ctx_loc = get_source_location_str(g_current_eval_context);
          } catch (...) {
            // Skip location if it crashes
          }
          fprintf(stderr, "\n  While evaluating");
          if (ctx_loc) {
            fprintf(stderr, " (at %s)", ctx_loc);
          }
          fprintf(stderr, ": ");
          try {
            print_ast_to_stderr(g_current_eval_context);
          } catch (...) {
            fprintf(stderr, "<unable to print - corrupted pointer>");
          }
        }
      } catch (...) {
        // If accessing context crashes, just skip it
      }
    }
    fprintf(stderr, "\n");
    
    // Print the evaluation call stack to show the evaluation path
    fprintf(stderr, "\n=== Evaluation Call Stack (showing how we got here) ===\n");
    if (g_eval_stack) {
      print_eval_stack();
    } else {
      fprintf(stderr, "Call stack is empty (error occurred at top level)\n");
    }
    fprintf(stderr, "=== End of Call Stack ===\n");
    
    // Print summary at the end for quick reference
    fprintf(stderr, "\n=== Error Summary ===\n");
    fprintf(stderr, "Error Type: not supported expression type: %s\n", type_name);
    if (loc_str) {
      fprintf(stderr, "Error Location: %s\n", loc_str);
    } else if (list_loc_str) {
      fprintf(stderr, "Error Location: %s\n", list_loc_str);
    } else {
      fprintf(stderr, "Error Location: <unknown> (check call stack above for details)\n");
    }
    fprintf(stderr, "Problematic Expression: ");
    if (expr) {
      try {
        print_ast_to_stderr(expr);
      } catch (...) {
        fprintf(stderr, "<unable to print - corrupted pointer>");
      }
    } else {
      fprintf(stderr, "<null>");
    }
    fprintf(stderr, "\n");
    
    // Flush stderr to ensure all output is visible before exiting
    fflush(stderr);
    
    // Use exit instead of abort to ensure output is flushed
    exit(1);
    return nullptr;  // Never reached, but satisfies compiler
  }
}

// Scheme eval function: (eval expr) -> evaluates expr in the current environment
SCM *scm_c_eval(SCM *expr) {
  // Evaluate the expression in the global environment
  return eval_with_env(&g_env, expr);
}

// defined?: Check if a symbol is defined in the current environment or module
// (defined? sym) -> #t if sym is defined, #f otherwise
SCM *scm_c_defined(SCM *sym) {
  if (!is_sym(sym)) {
    eval_error("defined?: expected symbol");
    return nullptr;  // Never reached, but satisfies compiler
  }
  
  SCM_Symbol *symbol = cast<SCM_Symbol>(sym);
  
  // 1. Check in lexical environment (search parent environments too)
  SCM_Environment::Entry *entry = scm_env_search_entry(&g_env, symbol, /*search_parent=*/true);
  if (entry) {
    return scm_bool_true();  // Found in environment
  }
  
  // 2. Check in current module
  SCM *current_mod = scm_current_module();
  if (current_mod && is_module(current_mod)) {
    SCM_Module *module = cast<SCM_Module>(current_mod);
    SCM *var = module_search_variable(module, symbol);
    if (var) {
      return scm_bool_true();  // Found in module (value can be #f, but it's still defined)
    }
  }
  
  return scm_bool_false();  // Not found
}

void init_eval() {
  scm_define_function("eval", 1, 0, 0, scm_c_eval);
  scm_define_function("defined?", 1, 0, 0, scm_c_defined);
}
