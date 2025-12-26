#pragma once

#include <assert.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Source location information
struct SCM_SourceLocation {
  const char *filename;
  int line;
  int column;
};

struct SCM {
  enum Type { NONE, NIL, LIST, PROC, CONT, FUNC, NUM, FLOAT, CHAR, BOOL, SYM, STR, MACRO, HASH_TABLE, RATIO, VECTOR, PORT, PROMISE, MODULE } type;

  void *value;
  SCM_SourceLocation *source_loc;  // Optional source location
};

// Forward declaration for type_error (implemented in eval.cc)
[[noreturn]] void type_error(SCM *data, const char *expected_type);

struct SCM_Environment;

struct SCM_List {
  SCM *data;
  SCM_List *next;
  bool is_dotted;  // true indicates this is the last node of a dotted pair (stores the cdr node)
};

struct SCM_Symbol {
  char *data;
  int len;
};

struct SCM_String {
  char *data;
  int len;
};

// Rational number (fraction) representation
// Stores a fraction as numerator/denominator
// Example: 3/4 would have numerator=3, denominator=4
// The fraction is always in simplified form (GCD reduced)
struct SCM_Rational {
  int64_t numerator;    // The numerator (top part) of the fraction
  int64_t denominator;  // The denominator (bottom part) of the fraction, always > 0
};

struct SCM_Procedure {
  SCM_Symbol *name;
  SCM_List *args;
  SCM_List *body;
  SCM_Environment *env;
};

struct SCM_Function {
  SCM_Symbol *name;
  int n_args;
  void *func_ptr;
  SCM *generic;
};

struct SCM_Continuation {
  jmp_buf cont_jump_buffer;
  size_t stack_len;
  void *stack_data;
  void *dst;
  SCM *arg;
  SCM_List *wind_chain;  // Saved wind chain when continuation was created
  SCM *saved_module;     // Saved current module when continuation was created
};

struct SCM_Macro {
  SCM_Symbol *name;
  SCM_Procedure *transformer;  // Macro transformer procedure
  SCM_Environment *env;        // Environment where macro was defined
};

struct SCM_HashTable {
  SCM **buckets;        // Array of buckets (each bucket is a list of (key . value) pairs)
  size_t capacity;      // Number of buckets
  size_t size;          // Number of entries in the table
};

struct SCM_Vector {
  SCM **elements;       // Array of SCM pointers
  size_t length;        // Number of elements
};

struct SCM_Promise {
  SCM *thunk;      // A zero-argument procedure representing the delayed computation
  SCM *value;      // Cached value after forcing; nullptr if not yet forced
  bool is_forced;  // Whether the promise has been forced
};

struct SCM_Module {
  SCM_HashTable *obarray;        // Local bindings hash table (symbol -> variable)
  SCM_List *uses;                // List of used modules
  SCM_Procedure *binder;         // Optional binding procedure (module symbol definep) -> variable | #f
  SCM_Procedure *eval_closure;   // Lookup strategy function (symbol definep) -> variable | #f
  SCM_Procedure *transformer;    // Syntax transformer (expr) -> expr
  SCM_List *name;                // Module name list, e.g. (guile-user)
  SCM_Symbol *kind;              // Module type: 'module, 'interface, 'directory
  SCM_Module *public_interface;  // Public interface module (points to another module object)
  SCM_List *exports;             // List of exported symbols (for public interface)
};

// Port types
enum PortType {
  PORT_FILE_INPUT,
  PORT_FILE_OUTPUT,
  PORT_STRING_INPUT,
  PORT_STRING_OUTPUT
};

struct SCM_Port {
  PortType port_type;
  bool is_input;         // true for input port, false for output port
  bool is_closed;        // true if port is closed
  FILE *file;            // For file ports
  char *string_data;     // For string input ports (read-only)
  int string_pos;        // Current position in string
  int string_len;        // Length of string
  char *output_buffer;   // For string output ports (growing buffer)
  int output_len;        // Current length of output buffer
  int output_capacity;   // Capacity of output buffer
};

struct SCM_Environment {
  struct Entry {
    char *key;
    SCM *value;
  };

  struct List {
    Entry *data;
    List *next;
  };

  List dummy;
  SCM_Environment *parent;
};

inline bool is_str(SCM *scm) {
  return scm->type == SCM::STR;
}

inline bool is_sym(SCM *scm) {
  return scm->type == SCM::SYM;
}

inline bool is_pair(SCM *scm) {
  return scm->type == SCM::LIST;
}

inline bool is_num(SCM *scm) {
  return scm->type == SCM::NUM;
}

inline bool is_float(SCM *scm) {
  return scm->type == SCM::FLOAT;
}

inline bool is_ratio(SCM *scm) {
  return scm->type == SCM::RATIO;
}

inline bool is_char(SCM *scm) {
  return scm->type == SCM::CHAR;
}

inline bool is_proc(SCM *scm) {
  return scm->type == SCM::PROC;
}

inline bool is_func(SCM *scm) {
  return scm->type == SCM::FUNC;
}

inline bool is_cont(SCM *scm) {
  return scm->type == SCM::CONT;
}

inline bool is_nil(SCM *scm) {
  return scm->type == SCM::NIL;
}

inline bool is_bool(SCM *scm) {
  return scm->type == SCM::BOOL;
}

// Check if a value is truthy (in Scheme, only #f is falsy, everything else is truthy)
inline bool is_truthy(SCM *scm) {
  if (!scm) return false;
  // Only #f is falsy in Scheme
  return !(is_bool(scm) && scm->value == nullptr);
}

// Check if a value is falsy (only #f is falsy in Scheme)
inline bool is_falsy(SCM *scm) {
  if (!scm) return true;
  // Only #f is falsy in Scheme
  return is_bool(scm) && scm->value == nullptr;
}

// Legacy functions - these assume the value is a boolean
// Use is_truthy/is_falsy for general truthiness checks
inline bool is_true(SCM *scm) {
  if (!scm || !is_bool(scm)) {
    type_error(scm, "boolean");
  }
  return scm->value;
}

inline bool is_false(SCM *scm) {
  if (!scm || !is_bool(scm)) {
    type_error(scm, "boolean");
  }
  return !scm->value;
}

inline bool is_none(SCM *scm) {
  return scm->type == SCM::NONE;
}

inline bool is_macro(SCM *scm) {
  return scm->type == SCM::MACRO;
}

inline bool is_hash_table(SCM *scm) {
  return scm->type == SCM::HASH_TABLE;
}

inline bool is_vector(SCM *scm) {
  return scm->type == SCM::VECTOR;
}

inline bool is_promise(SCM *scm) {
  return scm->type == SCM::PROMISE;
}

inline bool is_port(SCM *scm) {
  return scm->type == SCM::PORT;
}

inline bool is_module(SCM *scm) {
  return scm->type == SCM::MODULE;
}

SCM *create_sym(const char *data, int len);

// Source location functions (implemented in source_location.cc)
void set_source_location(SCM *scm, const char *filename, int line, int column);
void copy_source_location(SCM *dest, SCM *src);
void copy_source_location_recursive(SCM *dest, SCM *src);
const char *get_source_location_str(SCM *scm);

// Inline helper function
inline SCM_List make_list_dummy() {
  SCM_List dummy;
  dummy.data = nullptr;
  dummy.next = nullptr;
  dummy.is_dotted = false;
  return dummy;
}

inline SCM_List *make_list() {
  auto l = new SCM_List();
  l->data = nullptr;
  l->next = nullptr;
  l->is_dotted = false;
  return l;
}

inline SCM_List *make_list(SCM *data) {
  auto l = make_list();
  l->data = data;
  return l;
}

inline SCM_List *make_list(SCM *data1, SCM *data2) {
  auto l = make_list(data1);
  l->next = make_list(data2);
  return l;
}

inline SCM_List *make_list(SCM *data1, SCM *data2, SCM *data3) {
  auto l = make_list(data1);
  l->next = make_list(data2, data3);
  return l;
}

inline SCM_Procedure *make_proc(SCM_Symbol *name, SCM_List *args, SCM_List *body, SCM_Environment *env) {
  // name can be nullptr
  // assert(name);
  // args can be nullptr
  // assert(args);
  assert(body);
  assert(env);
  SCM_Procedure *proc = new SCM_Procedure();
  proc->name = name;
  proc->args = args;
  proc->body = body;
  proc->env = env;
  return proc;
}

inline SCM_Continuation *make_cont(size_t stack_len, void *stack_data) {
  auto cont = new SCM_Continuation();
  cont->stack_len = stack_len;
  cont->stack_data = stack_data;
  cont->wind_chain = nullptr;
  cont->saved_module = nullptr;
  return cont;
}

inline SCM_Environment *make_env(SCM_Environment *parent) {
  auto env = new SCM_Environment();
  env->parent = parent;
  env->dummy.data = nullptr;
  env->dummy.next = nullptr;
  return env;
}

SCM *scm_list1(SCM *arg1);
SCM *scm_list2(SCM *arg1, SCM *arg2);
SCM *scm_list3(SCM *arg1, SCM *arg2, SCM *arg3);
SCM *scm_list(SCM_List *args);
SCM *scm_cons(SCM *car_val, SCM *cdr_val);
SCM *scm_append(SCM_List *args);

SCM *scm_concat_list2(SCM *arg1, SCM *arg2);

SCM *scm_none();
SCM *scm_nil();
SCM *scm_bool_false();
SCM *scm_bool_true();

// Float number helper functions
// double <-> void* conversion helpers (for 64-bit systems)
inline void* double_to_ptr(double val) {
  union {
    double d;
    void *p;
  } u;
  u.d = val;
  return u.p;
}

inline double ptr_to_double(void *ptr) {
  union {
    double d;
    void *p;
  } u;
  u.p = ptr;
  return u.d;
}

// Create a float number
inline SCM *scm_from_double(double val) {
  SCM *scm = new SCM();
  scm->type = SCM::FLOAT;
  scm->value = double_to_ptr(val);
  scm->source_loc = nullptr;
  return scm;
}

// Convert SCM to double (handles NUM, FLOAT, and RATIO)
inline double scm_to_double(SCM *scm) {
  if (is_float(scm)) {
    return ptr_to_double(scm->value);
  } else if (is_num(scm)) {
    return (double)(int64_t)scm->value;
  } else if (is_ratio(scm)) {
    SCM_Rational *rat = (SCM_Rational *)scm->value;
    return (double)rat->numerator / (double)rat->denominator;
  }
  return 0.0;
}

// Character helper functions
// char <-> void* conversion helpers
inline void* char_to_ptr(char val) {
  return (void*)(uintptr_t)(unsigned char)val;
}

inline char ptr_to_char(void *ptr) {
  return (char)(uintptr_t)ptr;
}

// Create a character
inline SCM *scm_from_char(char val) {
  SCM *scm = new SCM();
  scm->type = SCM::CHAR;
  scm->value = char_to_ptr(val);
  scm->source_loc = nullptr;
  return scm;
}

// Convert SCM to char
inline char scm_to_char(SCM *scm) {
  if (is_char(scm)) {
    return ptr_to_char(scm->value);
  }
  return 0;
}
SCM *scm_sym_lambda();
SCM *scm_sym_set();
SCM *scm_sym_let();
SCM *scm_sym_letrec();
SCM *scm_sym_quote();
SCM *scm_sym_quasiquote();
SCM *scm_sym_unquote();
SCM *scm_sym_unquote_splicing();

template <typename T>
SCM *wrap(T *);

template <>
inline SCM *wrap(SCM_Procedure *proc) {
  assert(proc);
  auto data = new SCM();
  data->type = SCM::PROC;
  data->value = proc;
  data->source_loc = nullptr;
  return data;
}

template <>
inline SCM *wrap(SCM_Continuation *cont) {
  assert(cont);
  auto data = new SCM();
  data->type = SCM::CONT;
  data->value = cont;
  data->source_loc = nullptr;
  return data;
}

template <>
inline SCM *wrap(SCM_Function *func) {
  assert(func);
  auto data = new SCM();
  data->type = SCM::FUNC;
  data->value = func;
  data->source_loc = nullptr;
  return data;
}

template <>
inline SCM *wrap(SCM_List *l) {
  if (!l) {
    return scm_nil();
  }
  assert(l);
  auto data = new SCM();
  data->type = SCM::LIST;
  data->value = l;
  data->source_loc = nullptr;
  return data;
}

template <>
inline SCM *wrap(SCM_Symbol *sym) {
  assert(sym);
  auto data = new SCM();
  data->type = SCM::SYM;
  data->value = sym;
  data->source_loc = nullptr;
  return data;
}

template <typename T>
T *cast(SCM *);

template <>
inline SCM_List *cast<SCM_List>(SCM *data) {
  if (!data || (!is_pair(data) && !is_nil(data))) {
    type_error(data, "pair or nil");
  }
  auto l = (SCM_List *)data->value;
  return l;
}

template <>
inline SCM_String *cast<SCM_String>(SCM *data) {
  if (!data || !is_str(data)) {
    type_error(data, "string");
  }
  auto l = (SCM_String *)data->value;
  return l;
}

template <>
inline SCM_Symbol *cast<SCM_Symbol>(SCM *data) {
  if (!data || !is_sym(data)) {
    type_error(data, "symbol");
  }
  auto l = (SCM_Symbol *)data->value;
  return l;
}

template <>
inline SCM_Procedure *cast<SCM_Procedure>(SCM *data) {
  if (!data || !is_proc(data)) {
    type_error(data, "procedure");
  }
  auto l = (SCM_Procedure *)data->value;
  return l;
}

template <>
inline SCM_Function *cast<SCM_Function>(SCM *data) {
  if (!data || !is_func(data)) {
    type_error(data, "function");
  }
  auto l = (SCM_Function *)data->value;
  return l;
}

template <>
inline SCM_Continuation *cast<SCM_Continuation>(SCM *data) {
  if (!data || !is_cont(data)) {
    type_error(data, "continuation");
  }
  auto l = (SCM_Continuation *)data->value;
  return l;
}

template <>
inline SCM_Promise *cast<SCM_Promise>(SCM *data) {
  if (!data || !is_promise(data)) {
    type_error(data, "promise");
  }
  auto p = (SCM_Promise *)data->value;
  return p;
}

template <>
inline SCM *wrap(SCM_Macro *macro) {
  assert(macro);
  auto data = new SCM();
  data->type = SCM::MACRO;
  data->value = macro;
  data->source_loc = nullptr;
  return data;
}

template <>
inline SCM_HashTable *cast<SCM_HashTable>(SCM *data) {
  if (!data || !is_hash_table(data)) {
    type_error(data, "hash-table");
  }
  auto l = (SCM_HashTable *)data->value;
  return l;
}

template <>
inline SCM *wrap(SCM_HashTable *hash_table) {
  assert(hash_table);
  auto data = new SCM();
  data->type = SCM::HASH_TABLE;
  data->value = hash_table;
  data->source_loc = nullptr;
  return data;
}

template <>
inline SCM_Macro *cast<SCM_Macro>(SCM *data) {
  if (!data || !is_macro(data)) {
    type_error(data, "macro");
  }
  return (SCM_Macro *)data->value;
}

template <>
inline SCM_Rational *cast<SCM_Rational>(SCM *data) {
  if (!data || !is_ratio(data)) {
    type_error(data, "ratio");
  }
  return (SCM_Rational *)data->value;
}

template <>
inline SCM_Vector *cast<SCM_Vector>(SCM *data) {
  if (!data || !is_vector(data)) {
    type_error(data, "vector");
  }
  return (SCM_Vector *)data->value;
}

template <>
inline SCM_Port *cast<SCM_Port>(SCM *data) {
  if (!data || !is_port(data)) {
    type_error(data, "port");
  }
  return (SCM_Port *)data->value;
}

template <>
inline SCM_Module *cast<SCM_Module>(SCM *data) {
  if (!data || !is_module(data)) {
    type_error(data, "module");
  }
  return (SCM_Module *)data->value;
}

template <>
inline SCM *wrap(SCM_Vector *vec) {
  assert(vec);
  auto data = new SCM();
  data->type = SCM::VECTOR;
  data->value = vec;
  data->source_loc = nullptr;
  return data;
}

inline SCM *wrap(SCM_Module *module) {
  assert(module);
  auto data = new SCM();
  data->type = SCM::MODULE;
  data->value = module;
  data->source_loc = nullptr;
  return data;
}

inline SCM *car(SCM *data) {
  if (!data || !is_pair(data)) {
    type_error(data, "pair");
  }
  auto l = cast<SCM_List>(data);
  if (!l) {
    type_error(data, "pair");
  }
  return l->data;
}

inline SCM *cdr(SCM *data) {
  if (!data || !is_pair(data)) {
    type_error(data, "pair");
  }
  auto l = cast<SCM_List>(data);
  if (!l) {
    type_error(data, "pair");
  }
  if (l->next == nullptr) {
    return scm_nil();
  }
  // Check if this is a dotted pair (is_dotted flag is set on the cdr node)
  if (l->next->is_dotted) {
    // For dotted pair (a . b), return b directly (not wrapped as a list)
    return l->next->data;
  }
  auto new_data = new SCM();
  new_data->type = SCM::LIST;
  new_data->value = l->next;
  new_data->source_loc = nullptr;  // Initialize to nullptr
  // Copy source location from original list
  copy_source_location(new_data, data);
  return new_data;
}

inline SCM *cadr(SCM *data) {
  if (!data || !is_pair(data)) {
    type_error(data, "pair");
  }
  auto l = cast<SCM_List>(data);
  if (!l || !l->next) {
    type_error(data, "pair with at least 2 elements");
  }
  SCM *result = l->next->data;
  // Copy source location if result doesn't have one
  if (result && !result->source_loc && data->source_loc) {
    copy_source_location(result, data);
  }
  return result;
}

inline SCM *cddr(SCM *data) {
  return cdr(cdr(data));
}

inline SCM *caddr(SCM *data) {
  SCM *result = car(cdr(cdr(data)));
  // Copy source location if result doesn't have one
  if (result && !result->source_loc && data->source_loc) {
    copy_source_location(result, data);
  }
  return result;
}

template <typename F>
SCM_List *map(SCM_List *l, F f) {
  SCM_List dummay_list;
  auto it = &dummay_list;
  it->next = NULL;
  while (l) {
    it->next = new SCM_List();
    it->next->data = f(l->data);
    it->next->next = NULL;
    it = it->next;
    l = l->next;
  }
  return dummay_list.next;
}

template <typename F>
SCM *map(SCM *data, F f) {
  if (!data || !is_pair(data)) {
    type_error(data, "pair");
  }
  assert(is_pair(data));
  auto l = cast<SCM_List>(data);
  assert(l);
  auto new_l = map(l, f);
  if (new_l) {
    return wrap(new_l);
  }
  return scm_nil();
}

template <typename F>
SCM *reduce(F f, SCM *init_val, SCM_List *l) {
  auto ret = init_val;
  while (l) {
    ret = f(ret, l->data);
    l = l->next;
  }
  return ret;
}

/*
 * Functions in print.cc
 */
void print_ast(SCM *ast, bool write_mode = false);
void print_list(SCM_List *l);

/*
 * Functions in parse.cc
 */
SCM *parse(const char *s);
SCM_List *parse_file(const char *filename);

/*
 * Functions in repl.cc
 */
void repl();

/*
 * Functions in continuation.cc
 */
SCM *scm_make_continuation(int *first);
void scm_dynthrow(SCM *cont, SCM *args);

/*
 * Functions in let.cc
 */
SCM *expand_let(SCM *expr);
SCM *expand_letstar(SCM *expr);
SCM *expand_letrec(SCM *expr);

/*
 * Functions in environment.cc
 */
void scm_env_insert(SCM_Environment *env, SCM_Symbol *sym, SCM *value, bool search_parent = true);
SCM *scm_env_search(SCM_Environment *env, SCM_Symbol *sym);
SCM *scm_env_exist(SCM_Environment *env, SCM_Symbol *sym);
SCM_Environment::Entry *scm_env_search_entry(SCM_Environment *env, SCM_Symbol *sym, bool search_parent = true);

/*
 * Functions in eval.cc
 */
SCM *eval_with_env(SCM_Environment *env, SCM *ast);
SCM *eval(SCM *ast);
SCM_List *eval_list_with_env(SCM_Environment *env, SCM_List *l);

/*
 * Functions in apply.cc
 */
SCM *scm_c_apply(SCM_List *args);
void init_apply();

/*
 * Functions in eq.cc (internal comparison functions)
 */
bool _eq(SCM *lhs, SCM *rhs);
SCM *scm_c_is_eq(SCM *lhs, SCM *rhs);

/*
 * Functions in number.cc (internal comparison functions)
 */
bool _number_eq(SCM *lhs, SCM *rhs);

/*
 * Functions in predicate.cc
 */
SCM *scm_c_is_procedure(SCM *arg);
SCM *scm_c_is_boolean(SCM *arg);
SCM *scm_c_is_null(SCM *arg);
SCM *scm_c_is_pair(SCM *arg);

/*
 * Functions in symbol.cc
 */
SCM_Symbol *make_sym(const char *data);
SCM *scm_c_gensym();

/*
 * Functions in predicate.cc
 */
SCM *scm_c_not(SCM *arg);

/*
 * Functions in quasiquote.cc
 */
SCM *eval_quasiquote(SCM_Environment *env, SCM_List *l);

/*
 * Functions in list.cc
 */
SCM *scm_list1(SCM *arg1);
SCM *scm_list2(SCM *arg1, SCM *arg2);
SCM *scm_list3(SCM *arg1, SCM *arg2, SCM *arg3);
SCM *scm_list(SCM_List *args);
SCM *scm_cons(SCM *car_val, SCM *cdr_val);
SCM *scm_append(SCM_List *args);

/*
 * Macro
 */
#define SCM_FILENAME_LINENO()                                                                                          \
  print_basename(__BASE_FILE__);                                                                                       \
  printf(":%d ", __LINE__);

#define SCM_INIT_CONT(cont, base)                                                                                      \
  {                                                                                                                    \
    cont = new SCM_Continuation();                                                                                     \
    long __stack_top__;                                                                                                \
    cont->dst = &__stack_top__;                                                                                        \
    cont->stack_len = (long)base - (long)cont->dst;                                                                    \
    SCM_DEBUG_CONT("stack_top: %p\n", &__stack_top__);                                                                 \
    SCM_DEBUG_CONT("stack_len: %zu\n", cont->stack_len);                                                               \
    cont->stack_data = (long *)malloc(sizeof(long) * cont->stack_len);                                                 \
    memcpy((void *)cont->stack_data, (void *)cont->dst, sizeof(long) * cont->stack_len);                               \
  }                                                                                                                    \
  while (0)                                                                                                            \
    ;

#define SCM_APPLY_CONT(cont)                                                                                           \
  {                                                                                                                    \
    long __cur__;                                                                                                      \
    SCM_DEBUG_CONT("jump from %p to %p use %p\n", &__cur__, cont->dst, cont);                                          \
    memcpy(cont->dst, cont->stack_data, sizeof(long) * cont->stack_len);                                               \
    longjmp(cont->cont_jump_buffer, 1);                                                                                \
  }                                                                                                                    \
  while (0)                                                                                                            \
    ;

#define SCM_DEBUG(category, fmt, ...)                                                                                  \
  {                                                                                                                    \
    if (debug_enabled) {                                                                                               \
      printf(category);                                                                                                \
      printf(" ");                                                                                                     \
      SCM_FILENAME_LINENO();                                                                                           \
      printf(fmt, ##__VA_ARGS__);                                                                                      \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)                                                                                                            \
    ;
#define SCM_LOG_LEVEL_TRACE 1
#define SCM_LOG_LEVEL_DEBUG 2
#define SCM_LOG_LEVEL_INFO 3
#define SCM_LOG_LEVEL_ERROR 4
#define SCM_LOGGING(level, category, fmt, ...)                                                                         \
  {                                                                                                                    \
    if (debug_enabled || level >= SCM_LOG_LEVEL_ERROR) {                                                               \
      printf(category);                                                                                                \
      printf(" ");                                                                                                     \
      SCM_FILENAME_LINENO();                                                                                           \
      printf(fmt, ##__VA_ARGS__);                                                                                      \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)                                                                                                            \
    ;

#define SCM_DEBUG_CONT(fmt, ...) SCM_LOGGING(SCM_LOG_LEVEL_DEBUG, "[cont]", fmt, ##__VA_ARGS__)
#define SCM_DEBUG_SYMTBL(fmt, ...) SCM_LOGGING(SCM_LOG_LEVEL_DEBUG, "[symtbl]", fmt, ##__VA_ARGS__)
#define SCM_ERROR_SYMTBL(fmt, ...) SCM_LOGGING(SCM_LOG_LEVEL_ERROR, "[symtbl]", fmt, ##__VA_ARGS__)
#define SCM_DEBUG_TRANS(fmt, ...) SCM_LOGGING(SCM_LOG_LEVEL_DEBUG, "[trans]", fmt, ##__VA_ARGS__)
#define SCM_DEBUG_EVAL(fmt, ...) SCM_LOGGING(SCM_LOG_LEVEL_DEBUG, "[eval]", fmt, ##__VA_ARGS__)
#define SCM_ERROR_EVAL(fmt, ...) SCM_LOGGING(SCM_LOG_LEVEL_ERROR, "[eval]", fmt, ##__VA_ARGS__)

// #undef SCM_DEBUG_CONT
// #define SCM_DEBUG_CONT(...)
// #undef SCM_DEBUG_SYMTBL
// #define SCM_DEBUG_SYMTBL(...)
// #undef SCM_DEBUG_TRANS
// #define SCM_DEBUG_TRANS(...)

#define PRINT_ESP()                                                                                                    \
  {                                                                                                                    \
    int esp;                                                                                                           \
    int ebp;                                                                                                           \
    int eip;                                                                                                           \
    asm("movl %%esp, %0" : "=r"(esp));                                                                                 \
    asm("movl %%ebp, %0" : "=r"(ebp));                                                                                 \
    printf("esp %p\n", (void *)esp);                                                                                   \
    printf("ebp %p\n", (void *)ebp);                                                                                   \
  }                                                                                                                    \
  while (0)                                                                                                            \
    ;

extern bool debug_enabled;
void print_basename(const char *path);

/*
 * Initialize
 */
void init_scm();
void init_predicate();
void init_list();
void init_symbol();
void init_apply();
void init_map();
void init_number();
void init_continuation();
void init_eq();
void init_alist();
void init_char();
void init_string();
void init_port();
void init_exit();
void init_load();
void init_delay();
void init_eval();
void init_values();
void init_hash_table();
void init_procedure();
void init_vector();
void init_modules();

extern SCM_Environment g_env;
extern SCM_List *g_wind_chain;  // Global wind chain for dynamic-wind
extern SCM *g_root_module;  // Root module (pscm-user)
extern long *cont_base;  // Stack base pointer for continuations

/*
 * Hash table functions
 */
SCM *scm_c_make_hash_table(SCM *size_arg);
SCM *scm_c_hash_set_eq(SCM *table, SCM *key, SCM *value);
SCM *scm_c_hash_set_eqv(SCM *table, SCM *key, SCM *value);
SCM *scm_c_hash_set_equal(SCM *table, SCM *key, SCM *value);
SCM *scm_c_hash_ref_eq(SCM *table, SCM *key);
SCM *scm_c_hash_ref_eqv(SCM *table, SCM *key);
SCM *scm_c_hash_ref_equal(SCM *table, SCM *key);
SCM *scm_c_hash_get_handle_eq(SCM *table, SCM *key);
SCM *scm_c_hash_get_handle_eqv(SCM *table, SCM *key);
SCM *scm_c_hash_get_handle_equal(SCM *table, SCM *key);

/*
 * Module functions
 */
SCM *scm_make_module(SCM_List *name, int obarray_size);
SCM *scm_current_module();
SCM *scm_set_current_module(SCM *module);
SCM *scm_module_variable(SCM_Module *module, SCM_Symbol *sym, bool definep);
SCM *scm_resolve_module(SCM_List *name);
void scm_module_export(SCM_Module *module, SCM_Symbol *sym);

// Module helper functions (for internal use, but may be used by other modules)
SCM *module_obarray_lookup(SCM_Module *module, SCM_Symbol *sym);
SCM *module_search_variable(SCM_Module *module, SCM_Symbol *sym);
SCM_Module *module_find_variable_module(SCM_Module *module, SCM_Symbol *sym);

template <typename F>
SCM_Function *_create_func(const char *name, F func_ptr) {
  auto func = new SCM_Function();
  auto func_name = new SCM_Symbol();
  func_name->len = strlen(name);
  func_name->data = (char *)malloc(sizeof(char) * (func_name->len + 1));
  memcpy(func_name->data, name, func_name->len);
  func_name->data[func_name->len] = '\0';  // Ensure null termination
  func->name = func_name;
  func->func_ptr = (void *)func_ptr;
  return func;
}

template <typename F>
void scm_define_function(const char *name, int req, int opt, int rst, F func_ptr) {
  auto func = _create_func(name, func_ptr);
  func->n_args = req;
  auto data = wrap(func);
  scm_env_insert(&g_env, func->name, data, /*search_parent=*/false);
}

template <typename F>
void scm_define_generic_function(const char *name, F func_ptr, SCM *init_val) {
  auto func = _create_func(name, func_ptr);
  func->generic = init_val;
  func->n_args = -1;
  auto data = wrap(func);
  scm_env_insert(&g_env, func->name, data, /*search_parent=*/false);
}

template <typename F>
void scm_define_vararg_function(const char *name, F func_ptr) {
  auto func = _create_func(name, func_ptr);
  func->n_args = -2;  // Special value for variable argument functions
  func->generic = nullptr;
  auto data = wrap(func);
  scm_env_insert(&g_env, func->name, data, /*search_parent=*/false);
}

inline bool is_sym_val(SCM *scm, const char *val) {
  return is_sym(scm) && strcmp(cast<SCM_Symbol>(scm)->data, val) == 0;
}
