#include "pscm.h"
#include "eval.h"
#include "throw.h"
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

// Standard error key
SCM *g_error_key = nullptr;

// Global catch stack (shared between scm_c_catch and scm_throw)
static struct catch_info *g_catch_stack[100];
static int g_catch_stack_top = 0;

// Jump buffer and return value structure
struct jmp_buf_and_retval {
  jmp_buf buf;           // Jump buffer
  SCM *throw_tag;        // Thrown tag
  SCM *throw_args;       // Thrown arguments
  bool active;           // Whether this catch is active
};

// Catch information structure
struct catch_info {
  SCM *tag;              // Catch tag (symbol or #t)
  struct jmp_buf_and_retval *jbr;  // Jump buffer and return value
  scm_t_catch_handler handler;     // Handler function
  void *handler_data;     // Handler data
};

// Helper function to check if two tags match
static bool tags_match(SCM *tag, SCM *key) {
  // If tag is #t, it matches all keys
  if (is_bool(tag) && is_true(tag)) {
    return true;
  }
  
  // If both are symbols, compare by name
  if (is_sym(tag) && is_sym(key)) {
    SCM_Symbol *tag_sym = cast<SCM_Symbol>(tag);
    SCM_Symbol *key_sym = cast<SCM_Symbol>(key);
    return strcmp(tag_sym->data, key_sym->data) == 0;
  }
  
  // Otherwise, use pointer equality
  return tag == key;
}

// Core catch function
SCM *scm_c_catch(SCM *tag,
                 scm_t_catch_body body, void *body_data,
                 scm_t_catch_handler handler, void *handler_data) {
  struct jmp_buf_and_retval jbr;
  jbr.throw_tag = nullptr;
  jbr.throw_args = nullptr;
  jbr.active = false;
  
  // Allocate catch_info on stack (so it's protected from GC)
  struct catch_info info;
  info.tag = tag;
  info.jbr = &jbr;
  info.handler = handler;
  info.handler_data = handler_data;
  
  // Create a pair (tag . catch_info) to store in wind chain
  // We'll use a special marker to distinguish catch entries from wind entries
  // For simplicity, we'll store the catch_info pointer in a list
  // Format: (tag . (jbr_ptr . (handler_ptr . handler_data_ptr)))
  // Actually, we need to store pointers, so let's use a simpler approach:
  // Store catch_info pointer directly (it's on stack, so it's safe)
  
  // Create entry: (tag . catch_info_ptr)
  // We'll use a list where data is tag and next->data is catch_info pointer
  // But SCM_List stores SCM*, not void*. So we need another approach.
  // Let's create a special SCM object to hold the catch_info pointer.
  // Actually, for now, let's just store it in a global list temporarily.
  // Better: use a hash table or similar. But for simplicity, let's use a list.
  
  // For now, we'll use a simpler approach: store catch_info in a static array
  // and use an index. But that's not scalable.
  
  // Better approach: create a wrapper SCM object that holds the catch_info pointer
  // But we don't have a way to create custom SCM objects easily.
  
  // Simplest approach for now: use a global list of active catch_info structures
  // We'll add the catch_info to a list when entering catch, remove when exiting.
  // But we need to be careful about nested catches.
  
  // Actually, let's use a different approach: store catch_info in the wind chain
  // by creating a special marker. We can use a special symbol or a special list structure.
  
  if (g_catch_stack_top >= 100) {
    eval_error("catch: too many nested catches");
  }
  
  // Push catch_info onto stack
  g_catch_stack[g_catch_stack_top++] = &info;
  
  // Set up jump buffer
  if (setjmp(jbr.buf)) {
    // Exception was thrown - we're in the catch handler
    g_catch_stack_top--;  // Pop from stack
    
    SCM *throw_tag = jbr.throw_tag;
    SCM *throw_args = jbr.throw_args;
    
    // Call handler
    SCM *result = handler(handler_data, throw_tag, throw_args);
    return result;
  } else {
    // Normal execution - call body
    jbr.active = true;
    SCM *result = body(body_data);
    jbr.active = false;
    
    // Pop from stack
    g_catch_stack_top--;
    
    return result;
  }
}

// Throw exception
SCM *scm_throw(SCM *key, SCM *args) {
  // Find matching catch in the catch stack
  // Search from top to bottom (most recent catch first)
  for (int i = g_catch_stack_top - 1; i >= 0; i--) {
    struct catch_info *info = g_catch_stack[i];
    
    if (info && info->jbr && info->jbr->active && tags_match(info->tag, key)) {
      // Found matching catch - set throw info and jump
      info->jbr->throw_tag = key;
      info->jbr->throw_args = args;
      longjmp(info->jbr->buf, 1);
    }
  }
  
  // No matching catch found - call uncaught throw handler
  scm_uncaught_throw(key, args);
  return nullptr;  // Never reached
}

// Helper function to write to a port (for error messages)
static void write_to_port(SCM_Port *port, const char *str) {
  if (!port || port->is_closed) {
    fputs(str, stderr);
    return;
  }
  
  if (port->port_type == PORT_FILE_OUTPUT && port->file) {
    fputs(str, port->file);
  } else if (port->port_type == PORT_STRING_OUTPUT) {
    // Append to string output buffer
    int len = strlen(str);
    if (port->output_len + len + 1 >= port->output_capacity) {
      while (port->output_len + len + 1 >= port->output_capacity) {
        port->output_capacity *= 2;
      }
      port->output_buffer = (char*)realloc(port->output_buffer, port->output_capacity);
    }
    memcpy(port->output_buffer + port->output_len, str, len);
    port->output_len += len;
    port->output_buffer[port->output_len] = '\0';
  } else {
    fputs(str, stderr);
  }
}

// Helper function to print error message to error port
static void handler_message(void *handler_data, SCM *tag, SCM *args) {
  extern SCM *scm_current_error_port();
  SCM *error_port = scm_current_error_port();
  SCM_Port *port = cast<SCM_Port>(error_port);
  
  const char *prog_name = handler_data ? (const char *)handler_data : "pscm";
  
  // Build message string
  char msg[1024];
  int pos = 0;
  pos += snprintf(msg + pos, sizeof(msg) - pos, "%s: uncaught throw to ", prog_name);
  
  if (is_sym(tag)) {
    SCM_Symbol *sym = cast<SCM_Symbol>(tag);
    pos += snprintf(msg + pos, sizeof(msg) - pos, "'%s", sym->data);
  } else {
    // For non-symbol tags, use a simple representation
    pos += snprintf(msg + pos, sizeof(msg) - pos, "<non-symbol>");
  }
  
  pos += snprintf(msg + pos, sizeof(msg) - pos, ": ");
  
  // For args, we'll print a simplified version
  if (args) {
    if (is_pair(args)) {
      SCM_List *args_list = cast<SCM_List>(args);
      if (args_list->data && is_str(args_list->data)) {
        SCM_String *s = cast<SCM_String>(args_list->data);
        pos += snprintf(msg + pos, sizeof(msg) - pos, "%s", s->data);
      } else {
        pos += snprintf(msg + pos, sizeof(msg) - pos, "<args>");
      }
    } else {
      pos += snprintf(msg + pos, sizeof(msg) - pos, "<args>");
    }
  } else {
    pos += snprintf(msg + pos, sizeof(msg) - pos, "()");
  }
  
  pos += snprintf(msg + pos, sizeof(msg) - pos, "\n");
  
  write_to_port(port, msg);
  
  // Flush the port
  if (port->port_type == PORT_FILE_OUTPUT && port->file) {
    fflush(port->file);
  }
  
  // Print call stack if available
  if (g_eval_stack) {
    write_to_port(port, "\n=== Evaluation Call Stack ===\n");
    // For call stack, we'll use stderr for now (can be improved later)
    print_eval_stack();
    write_to_port(port, "=== End of Call Stack ===\n");
    if (port->port_type == PORT_FILE_OUTPUT && port->file) {
      fflush(port->file);
    }
  }
}

// Handle by message without exiting (unless quit tag)
SCM *scm_handle_by_message_noexit(void *handler_data, SCM *tag, SCM *args) {
  // Check if tag is "quit"
  if (is_sym(tag)) {
    SCM_Symbol *tag_sym = cast<SCM_Symbol>(tag);
    if (strcmp(tag_sym->data, "quit") == 0) {
      // Extract exit status from args (first element if present)
      int exit_status = 0;
      if (args && is_pair(args)) {
        SCM_List *args_list = cast<SCM_List>(args);
        if (args_list->data && is_num(args_list->data)) {
          exit_status = (int)(int64_t)args_list->data->value;
        }
      }
      exit(exit_status);
    }
  }
  
  // Print message but don't exit
  handler_message(handler_data, tag, args);
  
  return scm_bool_false();
}

// Uncaught throw handler
[[noreturn]] void scm_uncaught_throw(SCM *key, SCM *args) {
  handler_message(nullptr, key, args);
  // Force output before exiting
  extern SCM *scm_current_error_port();
  extern SCM *scm_force_output(SCM_List *args);
  SCM *error_port = scm_current_error_port();
  SCM_List force_args;
  force_args.data = error_port;
  force_args.next = nullptr;
  force_args.is_dotted = false;
  scm_force_output(&force_args);
  exit(1);
}

// Throw with noreturn flag (similar to scm_throw but for lazy-catch)
// The noreturn flag is currently unused but kept for API compatibility
SCM *scm_ithrow(SCM *key, SCM *args, int noreturn) {
  (void)noreturn;  // Unused for now
  return scm_throw(key, args);
}

// Body function for Scheme catch (thunk)
static SCM *scm_body_thunk(void *body_data) {
  SCM *thunk = (SCM *)body_data;
  if (!is_proc(thunk)) {
    eval_error("catch: thunk must be a procedure");
  }
  SCM_Procedure *proc = cast<SCM_Procedure>(thunk);
  // Use the procedure's environment if available, otherwise use g_env
  // The procedure's environment captures the lexical scope where it was defined
  SCM_Environment *env = proc->env ? proc->env : &g_env;
  return apply_procedure(env, proc, nullptr);
}

// Handler function for Scheme catch
static SCM *scm_handle_by_proc(void *handler_data, SCM *tag, SCM *args) {
  SCM *handler = (SCM *)handler_data;
  if (!is_proc(handler)) {
    eval_error("catch: handler must be a procedure");
  }
  SCM_Procedure *proc = cast<SCM_Procedure>(handler);
  
  // Build argument list: (tag . args)
  SCM_List *arg_list = make_list(tag);
  if (args) {
    if (is_pair(args)) {
      SCM_List *args_list = cast<SCM_List>(args);
      SCM_List *current = args_list;
      SCM_List *tail = arg_list;
      while (current) {
        tail->next = make_list(current->data);
        tail = tail->next;
        if (!current->next) break;
        current = current->next;
      }
    } else {
      arg_list->next = make_list(args);
    }
  }
  
  // Use apply_procedure_with_values because arguments are already evaluated
  return apply_procedure_with_values(proc->env, proc, arg_list);
}

// Scheme-callable catch function
SCM *scm_c_catch_scheme(SCM_List *args) {
  if (!args || !args->next || !args->next->next) {
    eval_error("catch: expected 3 arguments (tag thunk handler)");
  }
  
  // Arguments are already evaluated by eval_with_func
  SCM *tag = args->data;
  SCM *thunk = args->next->data;
  SCM *handler = args->next->next->data;
  
  // Verify tag is symbol or #t
  if (!is_sym(tag) && !(is_bool(tag) && is_true(tag))) {
    eval_error("catch: tag must be a symbol or #t");
  }
  
  // Verify thunk and handler are procedures
  if (!is_proc(thunk)) {
    eval_error("catch: thunk must be a procedure");
  }
  if (!is_proc(handler)) {
    eval_error("catch: handler must be a procedure");
  }
  
  return scm_c_catch(tag,
                     scm_body_thunk, thunk,
                     scm_handle_by_proc, handler);
}

// Scheme-callable throw function
SCM *scm_c_throw_scheme(SCM_List *args) {
  if (!args || !args->data) {
    eval_error("throw: expected at least 1 argument (key . args)");
  }
  
  // Arguments are already evaluated by eval_with_func
  SCM *key = args->data;
  
  // Verify key is a symbol
  if (!is_sym(key)) {
    eval_error("throw: key must be a symbol");
  }
  
  // Build args list from remaining arguments (already evaluated)
  SCM_List *throw_args = nullptr;
  if (args->next) {
    SCM_List *current = args->next;
    SCM_List dummy = make_list_dummy();
    SCM_List *tail = &dummy;
    while (current) {
      tail->next = make_list(current->data);
      tail = tail->next;
      if (!current->next) break;
      current = current->next;
    }
    throw_args = dummy.next;
  }
  
  SCM *throw_args_wrapped = throw_args ? wrap(throw_args) : scm_nil();
  
  return scm_throw(key, throw_args_wrapped);
}

// Initialize throw system
void init_throw() {
  // Create standard error key
  g_error_key = wrap(make_sym("error"));
  
  // Register Scheme functions
  scm_define_vararg_function("catch", scm_c_catch_scheme);
  scm_define_vararg_function("throw", scm_c_throw_scheme);
}

