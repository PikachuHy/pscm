#pragma once

#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>

// Forward declarations
struct SCM_Smob;

// Source location information
struct SCM_SourceLocation {
  const char *filename;
  int line;
  int column;
};

struct SCM {
  enum Type {
    NONE,
    NIL,
    LIST,
    PROC,
    CONT,
    FUNC,
    NUM,
    FLOAT,
    CHAR,
    BOOL,
    SYM,
    STR,
    MACRO,
    HASH_TABLE,
    RATIO,
    VECTOR,
    PORT,
    PROMISE,
    MODULE,
    SMOB,
    VARIABLE
  } type;

  void *value;
  SCM_SourceLocation *source_loc; // Optional source location
};

// Forward declaration for type_error (implemented in eval.cc)
void type_error(SCM *data, const char *expected_type);

struct SCM_Environment;

struct SCM_List {
  SCM *data;
  SCM_List *next;
  bool is_dotted; // true indicates this is the last node of a dotted pair (stores the cdr node)
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
  int64_t numerator;   // The numerator (top part) of the fraction
  int64_t denominator; // The denominator (bottom part) of the fraction, always > 0
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
  SCM_List *wind_chain; // Saved wind chain when continuation was created
  SCM *saved_module;    // Saved current module when continuation was created
  long *stack_src;      // Saved stack source address at capture time (cont_base - stack_size)
};

struct SCM_Macro {
  SCM_Symbol *name;
  SCM_Procedure *transformer; // Macro transformer procedure
  SCM_Environment *env;       // Environment where macro was defined
};

struct SCM_HashTable {
  SCM **buckets;   // Array of buckets (each bucket is a list of (key . value) pairs)
  size_t capacity; // Number of buckets
  size_t size;     // Number of entries in the table
};

struct SCM_Vector {
  SCM **elements; // Array of SCM pointers
  size_t length;  // Number of elements
};

struct SCM_Promise {
  SCM *thunk;     // A zero-argument procedure representing the delayed computation
  SCM *value;     // Cached value after forcing; nullptr if not yet forced
  bool is_forced; // Whether the promise has been forced
};

struct SCM_Module {
  SCM_HashTable *obarray;       // Local bindings hash table (symbol -> variable)
  SCM_List *uses;               // List of used modules
  SCM_Procedure *binder;        // Optional binding procedure (module symbol definep) -> variable | #f
  SCM_Procedure *eval_closure;  // Lookup strategy function (symbol definep) -> variable | #f
  SCM_Procedure *transformer;   // Syntax transformer (expr) -> expr
  SCM_List *name;               // Module name list, e.g. (guile-user)
  SCM_Symbol *kind;             // Module type: 'module, 'interface, 'directory
  SCM_Module *public_interface; // Public interface module (points to another module object)
  SCM_List *exports;            // List of exported symbols (for public interface)
  SCM_List *autoload_specs;     // List of (mod-name . sym-list) for autoload
};

// Variable object (wraps a value, similar to Guile's variable cells)
struct SCM_Variable {
  SCM *value;  // The value stored in the variable (can be nullptr for unbound)
};

// Port types
enum PortType { PORT_FILE_INPUT, PORT_FILE_OUTPUT, PORT_STRING_INPUT, PORT_STRING_OUTPUT, PORT_SOFT };

struct SCM_Port {
  PortType port_type;
  bool is_input;       // true for input port, false for output port
  bool is_closed;      // true if port is closed
  FILE *file;          // For file ports
  char *string_data;   // For string input ports (read-only)
  int string_pos;      // Current position in string
  int string_len;      // Length of string
  char *output_buffer; // For string output ports (growing buffer)
  int output_len;      // Current length of output buffer
  int output_capacity; // Capacity of output buffer
  // For soft ports: vector of procedures
  // [0] = procedure accepting one character for output
  // [1] = procedure accepting a string for output
  // [2] = thunk for flushing output
  // [3] = thunk for getting one character
  // [4] = thunk for closing port
  // [5] = (optional) thunk for computing number of characters available
  SCM_Vector *soft_procedures;  // Vector of procedures for soft port
  char *soft_modes;              // Modes string for soft port ("r", "w", "rw")
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

// Print state structure (compatible with Guile 1.8)
// Used to track state during printing, including circular reference detection
struct scm_print_state {
  SCM *handle;                    // Handle to the print state object (can be nullptr)
  int revealed;                   // Has the state escaped to Scheme? (0 = false, 1 = true)
  unsigned long writingp;         // Writing mode? (0 = display, 1 = write)
  unsigned long fancyp;           // Fancy printing? (for pretty-printing)
  unsigned long level;            // Max level (for truncation)
  unsigned long length;           // Max number of objects per level
  SCM *hot_ref;                   // Hot reference (for circular reference detection)
  unsigned long list_offset;      // List offset
  unsigned long top;              // Top of reference stack
  unsigned long ceiling;          // Max size of reference stack
  SCM_Vector *ref_vect;           // Stack of references used during circular reference detection
  SCM_List *highlight_objects;   // List of objects to be highlighted
};
