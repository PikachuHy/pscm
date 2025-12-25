#include "pscm.h"
#include "eval.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

// Forward declaration
SCM *parse(const char *s);

// Helper function to create a port SCM object
static SCM *wrap_port(SCM_Port *port) {
  SCM *scm = new SCM();
  scm->type = SCM::PORT;
  scm->value = port;
  scm->source_loc = nullptr;
  return scm;
}

// EOF object singleton
static SCM *g_eof_object = nullptr;

SCM *scm_eof_object() {
  if (!g_eof_object) {
    g_eof_object = new SCM();
    g_eof_object->type = SCM::NONE;  // Use NONE as EOF marker
    g_eof_object->value = (void*)0xDEADBEEF;  // Special marker value
    g_eof_object->source_loc = nullptr;
  }
  return g_eof_object;
}

bool is_eof_object(SCM *obj) {
  return obj == g_eof_object || 
         (obj && obj->type == SCM::NONE && obj->value == (void*)0xDEADBEEF);
}

// open-input-file: Open a file for input
SCM *scm_c_open_input_file(SCM *filename) {
  if (!is_str(filename)) {
    eval_error("open-input-file: expected string");
    return nullptr;
  }
  
  SCM_String *s = cast<SCM_String>(filename);
  FILE *f = fopen(s->data, "r");
  if (!f) {
    eval_error("open-input-file: cannot open file: %s", s->data);
    return nullptr;
  }
  
  SCM_Port *port = new SCM_Port();
  port->port_type = PORT_FILE_INPUT;
  port->is_input = true;
  port->is_closed = false;
  port->file = f;
  port->string_data = nullptr;
  port->string_pos = 0;
  port->string_len = 0;
  port->output_buffer = nullptr;
  port->output_len = 0;
  port->output_capacity = 0;
  
  return wrap_port(port);
}

// open-output-file: Open a file for output
SCM *scm_c_open_output_file(SCM *filename) {
  if (!is_str(filename)) {
    eval_error("open-output-file: expected string");
    return nullptr;
  }
  
  SCM_String *s = cast<SCM_String>(filename);
  FILE *f = fopen(s->data, "w");
  if (!f) {
    eval_error("open-output-file: cannot open file: %s", s->data);
    return nullptr;
  }
  
  SCM_Port *port = new SCM_Port();
  port->port_type = PORT_FILE_OUTPUT;
  port->is_input = false;
  port->is_closed = false;
  port->file = f;
  port->string_data = nullptr;
  port->string_pos = 0;
  port->string_len = 0;
  port->output_buffer = nullptr;
  port->output_len = 0;
  port->output_capacity = 0;
  
  return wrap_port(port);
}

// close-input-port: Close an input port
SCM *scm_c_close_input_port(SCM *port) {
  if (!is_port(port)) {
    eval_error("close-input-port: expected port");
    return nullptr;
  }
  
  SCM_Port *p = cast<SCM_Port>(port);
  if (!p->is_input) {
    eval_error("close-input-port: expected input port");
    return nullptr;
  }
  
  if (p->is_closed) {
    return scm_none();
  }
  
  if (p->file) {
    fclose(p->file);
    p->file = nullptr;
  }
  p->is_closed = true;
  
  return scm_none();
}

// close-output-port: Close an output port
SCM *scm_c_close_output_port(SCM *port) {
  if (!is_port(port)) {
    eval_error("close-output-port: expected port");
    return nullptr;
  }
  
  SCM_Port *p = cast<SCM_Port>(port);
  if (p->is_input) {
    eval_error("close-output-port: expected output port");
    return nullptr;
  }
  
  if (p->is_closed) {
    return scm_none();
  }
  
  if (p->file) {
    fclose(p->file);
    p->file = nullptr;
  } else if (p->output_buffer) {
    // For string output ports, buffer is kept for get-output-string
  }
  p->is_closed = true;
  
  return scm_none();
}

// open-input-string: Create an input port from a string
SCM *scm_c_open_input_string(SCM *str) {
  if (!is_str(str)) {
    eval_error("open-input-string: expected string");
    return nullptr;
  }
  
  SCM_String *s = cast<SCM_String>(str);
  SCM_Port *port = new SCM_Port();
  port->port_type = PORT_STRING_INPUT;
  port->is_input = true;
  port->is_closed = false;
  port->file = nullptr;
  port->string_data = new char[s->len + 1];
  memcpy(port->string_data, s->data, s->len);
  port->string_data[s->len] = '\0';
  port->string_pos = 0;
  port->string_len = s->len;
  port->output_buffer = nullptr;
  port->output_len = 0;
  port->output_capacity = 0;
  
  return wrap_port(port);
}

// open-output-string: Create an output port that writes to a string
SCM *scm_c_open_output_string() {
  SCM_Port *port = new SCM_Port();
  port->port_type = PORT_STRING_OUTPUT;
  port->is_input = false;
  port->is_closed = false;
  port->file = nullptr;
  port->string_data = nullptr;
  port->string_pos = 0;
  port->string_len = 0;
  port->output_buffer = (char*)malloc(256);
  port->output_len = 0;
  port->output_capacity = 256;
  port->output_buffer[0] = '\0';
  
  return wrap_port(port);
}

// Helper function to create string from C string (needed by get-output-string)
static SCM *scm_from_c_string_port(const char *data, int len) {
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

// get-output-string: Get the string from an output string port
SCM *scm_c_get_output_string(SCM *port) {
  if (!is_port(port)) {
    eval_error("get-output-string: expected port");
    return nullptr;
  }
  
  SCM_Port *p = cast<SCM_Port>(port);
  if (p->port_type != PORT_STRING_OUTPUT) {
    eval_error("get-output-string: expected output string port");
    return nullptr;
  }
  
  if (!p->output_buffer) {
    return scm_from_c_string_port("", 0);
  }
  
  return scm_from_c_string_port(p->output_buffer, p->output_len);
}

// Helper function to read a character from a port
static int read_char_from_port(SCM_Port *port) {
  if (port->is_closed) {
    return EOF;
  }
  
  if (port->port_type == PORT_FILE_INPUT) {
    if (!port->file) {
      return EOF;
    }
    return fgetc(port->file);
  } else if (port->port_type == PORT_STRING_INPUT) {
    if (port->string_pos >= port->string_len) {
      return EOF;
    }
    return (unsigned char)port->string_data[port->string_pos++];
  }
  
  return EOF;
}

// Helper function to unread a character (for peek)
static void unread_char_from_port(SCM_Port *port) {
  if (port->is_closed) {
    return;
  }
  
  if (port->port_type == PORT_FILE_INPUT) {
    if (port->file) {
      ungetc(fgetc(port->file), port->file);  // This is a hack, we need to track last char
    }
  } else if (port->port_type == PORT_STRING_INPUT) {
    if (port->string_pos > 0) {
      port->string_pos--;
    }
  }
}

// Helper function to peek a character from a port
static int peek_char_from_port(SCM_Port *port) {
  if (port->is_closed) {
    return EOF;
  }
  
  if (port->port_type == PORT_FILE_INPUT) {
    if (!port->file) {
      return EOF;
    }
    int ch = fgetc(port->file);
    if (ch != EOF) {
      ungetc(ch, port->file);
    }
    return ch;
  } else if (port->port_type == PORT_STRING_INPUT) {
    if (port->string_pos >= port->string_len) {
      return EOF;
    }
    return (unsigned char)port->string_data[port->string_pos];
  }
  
  return EOF;
}

// Helper function to write a character to a port (exported for use in string.cc)
void write_char_to_port_string(SCM_Port *port, char ch) {
  if (port->is_closed) {
    return;
  }
  
  if (port->port_type == PORT_FILE_OUTPUT) {
    if (port->file) {
      fputc(ch, port->file);
    }
  } else if (port->port_type == PORT_STRING_OUTPUT) {
    if (port->output_len + 1 >= port->output_capacity) {
      port->output_capacity *= 2;
      port->output_buffer = (char*)realloc(port->output_buffer, port->output_capacity);
    }
    port->output_buffer[port->output_len++] = ch;
    port->output_buffer[port->output_len] = '\0';
  }
}

// Helper function to read a complete expression from port
// This reads characters until we have a complete expression
static SCM *read_expr_from_port(SCM_Port *port) {
  if (port->is_closed) {
    return scm_eof_object();
  }
  
  // Build a buffer with the expression
  // Use dynamic allocation for large expressions
  size_t buffer_size = 4096;
  char *buffer = (char *)malloc(buffer_size);
  if (!buffer) {
    eval_error("read: out of memory");
    return scm_eof_object();
  }
  int pos = 0;
  int paren_depth = 0;
  bool in_string = false;
  bool escaped = false;
  bool started = false;
  
  // Skip leading whitespace and comments
  while (true) {
    int ch = peek_char_from_port(port);
    if (ch == EOF) {
      if (!started) {
        return scm_eof_object();
      }
      break;
    }
    
    // Skip whitespace
    if (isspace(ch)) {
      read_char_from_port(port);  // Consume whitespace
      continue;
    }
    
    // Skip comments (lines starting with ;)
    if (ch == ';') {
      // Read until end of line
      while (true) {
        int comment_ch = read_char_from_port(port);
        if (comment_ch == EOF || comment_ch == '\n' || comment_ch == '\r') {
          // Skip the newline if we read it
          if (comment_ch == '\r') {
            int next_ch = peek_char_from_port(port);
            if (next_ch == '\n') {
              read_char_from_port(port);
            }
          }
          break;
        }
      }
      continue;
    }
    
    // Found non-whitespace, non-comment character
    started = true;
    break;
  }
  
  if (!started) {
    return scm_eof_object();
  }
  
  // Read until we have a complete expression
  while (pos < (int)buffer_size - 1) {
    int ch = read_char_from_port(port);
    if (ch == EOF) {
      if (paren_depth == 0 && !in_string) {
        break;
      }
      // Incomplete expression
      free(buffer);
      return scm_eof_object();
    }
    
    // Resize buffer if needed
    if (pos >= (int)buffer_size - 1) {
      buffer_size *= 2;
      char *new_buffer = (char *)realloc(buffer, buffer_size);
      if (!new_buffer) {
        free(buffer);
        eval_error("read: out of memory");
        return scm_eof_object();
      }
      buffer = new_buffer;
    }
    
    buffer[pos++] = (char)ch;
    
    if (escaped) {
      escaped = false;
      continue;
    }
    
    if (ch == '\\' && in_string) {
      escaped = true;
      continue;
    }
    
    if (ch == '"') {
      in_string = !in_string;
      continue;
    }
    
    if (in_string) {
      continue;
    }
    
    if (ch == '(') {
      paren_depth++;
    } else if (ch == ')') {
      paren_depth--;
      if (paren_depth < 0) {
        // Too many closing parens
        break;
      }
      if (paren_depth == 0) {
        // Complete expression
        break;
      }
    } else if (paren_depth == 0 && (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r')) {
      // End of expression at top level
      pos--;  // Don't include the whitespace
      break;
    }
  }
  
  buffer[pos] = '\0';
  
  // Parse the expression
  SCM *result = parse(buffer);
  
  // Free the buffer
  free(buffer);
  
  return result;
}

// read-char: Read a character from a port
SCM *scm_c_read_char(SCM_List *args) {
  SCM_Port *port;
  SCM *port_arg = nullptr;
  
  if (args && args->data) {
    port_arg = args->data;
  }
  
  if (!port_arg || is_nil(port_arg)) {
    // Use stdin
    eval_error("read-char: stdin port not yet implemented");
    return nullptr;
  }
  
  if (!is_port(port_arg)) {
    eval_error("read-char: expected port");
    return nullptr;
  }
  
  port = cast<SCM_Port>(port_arg);
  if (!port->is_input) {
    eval_error("read-char: expected input port");
    return nullptr;
  }
  
  int ch = read_char_from_port(port);
  if (ch == EOF) {
    return scm_eof_object();
  }
  
  return scm_from_char((char)ch);
}

// peek-char: Peek at the next character from a port
SCM *scm_c_peek_char(SCM_List *args) {
  SCM_Port *port;
  SCM *port_arg = nullptr;
  
  if (args && args->data) {
    port_arg = args->data;
  }
  
  if (!port_arg || is_nil(port_arg)) {
    // Use stdin
    eval_error("peek-char: stdin port not yet implemented");
    return nullptr;
  }
  
  if (!is_port(port_arg)) {
    eval_error("peek-char: expected port");
    return nullptr;
  }
  
  port = cast<SCM_Port>(port_arg);
  if (!port->is_input) {
    eval_error("peek-char: expected input port");
    return nullptr;
  }
  
  int ch = peek_char_from_port(port);
  if (ch == EOF) {
    return scm_eof_object();
  }
  
  return scm_from_char((char)ch);
}

// read: Read a Scheme object from a port
SCM *scm_c_read(SCM_List *args) {
  SCM_Port *port;
  SCM *port_arg = nullptr;
  
  if (args && args->data) {
    port_arg = args->data;
  }
  
  if (!port_arg || is_nil(port_arg)) {
    // Use stdin - for now, use parse from string
    eval_error("read: stdin port not yet implemented");
    return nullptr;
  }
  
  if (!is_port(port_arg)) {
    eval_error("read: expected port");
    return nullptr;
  }
  
  port = cast<SCM_Port>(port_arg);
  if (!port->is_input) {
    eval_error("read: expected input port");
    return nullptr;
  }
  
  // Use the helper function to read a complete expression
  return read_expr_from_port(port);
}

// eof-object?: Check if an object is the EOF object
SCM *scm_c_is_eof_object(SCM *obj) {
  return is_eof_object(obj) ? scm_bool_true() : scm_bool_false();
}

// input-port?: Check if an object is an input port
SCM *scm_c_is_input_port(SCM *obj) {
  if (!is_port(obj)) {
    return scm_bool_false();
  }
  SCM_Port *port = cast<SCM_Port>(obj);
  return port->is_input ? scm_bool_true() : scm_bool_false();
}

// output-port?: Check if an object is an output port
SCM *scm_c_is_output_port(SCM *obj) {
  if (!is_port(obj)) {
    return scm_bool_false();
  }
  SCM_Port *port = cast<SCM_Port>(obj);
  return port->is_input ? scm_bool_false() : scm_bool_true();
}

// write-char: Write a character to a port
SCM *scm_c_write_char(SCM_List *args) {
  if (!args || !args->data) {
    eval_error("write-char: requires at least 1 argument");
    return nullptr;
  }
  
  SCM *char_arg = args->data;
  SCM *port_arg = nullptr;
  if (args->next && args->next->data && !is_nil(args->next->data)) {
    port_arg = args->next->data;
  }
  
  if (!is_char(char_arg)) {
    eval_error("write-char: first argument must be a character");
    return nullptr;
  }
  
  char ch = ptr_to_char(char_arg->value);
  
  if (port_arg) {
    if (!is_port(port_arg)) {
      eval_error("write-char: second argument must be a port");
      return nullptr;
    }
    SCM_Port *port = cast<SCM_Port>(port_arg);
    if (port->is_input) {
      eval_error("write-char: expected output port");
      return nullptr;
    }
    write_char_to_port_string(port, ch);
  } else {
    // Write to stdout
    putchar(ch);
  }
  
  return scm_none();
}

// char-ready?: Check if a character is ready to be read
SCM *scm_c_char_ready(SCM_List *args) {
  SCM_Port *port;
  SCM *port_arg = nullptr;
  
  if (args && args->data) {
    port_arg = args->data;
  }
  
  if (!port_arg || is_nil(port_arg)) {
    // For stdin, we can't really check, return true
    return scm_bool_true();
  }
  
  if (!is_port(port_arg)) {
    eval_error("char-ready?: expected port");
    return nullptr;
  }
  
  port = cast<SCM_Port>(port_arg);
  if (!port->is_input) {
    eval_error("char-ready?: expected input port");
    return nullptr;
  }
  
  if (port->is_closed) {
    return scm_bool_false();
  }
  
  if (port->port_type == PORT_FILE_INPUT) {
    if (!port->file) {
      return scm_bool_false();
    }
    // Check if there's data available
    int ch = peek_char_from_port(port);
    return (ch != EOF) ? scm_bool_true() : scm_bool_false();
  } else if (port->port_type == PORT_STRING_INPUT) {
    return (port->string_pos < port->string_len) ? scm_bool_true() : scm_bool_false();
  }
  
  return scm_bool_false();
}

// call-with-input-file: Open file, call proc, then close file
SCM *scm_c_call_with_input_file(SCM_List *args) {
  if (!args || !args->data || !args->next || !args->next->data) {
    eval_error("call-with-input-file: requires 2 arguments (filename proc)");
    return nullptr;
  }
  
  SCM *filename = args->data;
  SCM *proc = args->next->data;
  
  if (!is_str(filename)) {
    eval_error("call-with-input-file: first argument must be a string");
    return nullptr;
  }
  
  if (!is_proc(proc) && !is_func(proc)) {
    eval_error("call-with-input-file: second argument must be a procedure");
    return nullptr;
  }
  
  // Open the file
  SCM *port = scm_c_open_input_file(filename);
  if (!port) {
    return nullptr;
  }
  
  // Call the procedure with the port
  SCM_List proc_args;
  proc_args.data = port;
  proc_args.next = nullptr;
  proc_args.is_dotted = false;
  
  SCM *result;
  if (is_proc(proc)) {
    SCM_Procedure *proc_obj = cast<SCM_Procedure>(proc);
    result = apply_procedure(g_env.parent ? g_env.parent : &g_env, proc_obj, &proc_args);
  } else {
    SCM_Function *func_obj = cast<SCM_Function>(proc);
    SCM_List func_call;
    func_call.data = proc;
    func_call.next = &proc_args;
    result = eval_with_func(func_obj, &func_call);
  }
  
  // Close the port
  scm_c_close_input_port(port);
  
  return result;
}

// call-with-output-file: Open file, call proc, then close file
SCM *scm_c_call_with_output_file(SCM_List *args) {
  if (!args || !args->data || !args->next || !args->next->data) {
    eval_error("call-with-output-file: requires 2 arguments (filename proc)");
    return nullptr;
  }
  
  SCM *filename = args->data;
  SCM *proc = args->next->data;
  
  if (!is_str(filename)) {
    eval_error("call-with-output-file: first argument must be a string");
    return nullptr;
  }
  
  if (!is_proc(proc) && !is_func(proc)) {
    eval_error("call-with-output-file: second argument must be a procedure");
    return nullptr;
  }
  
  // Open the file
  SCM *port = scm_c_open_output_file(filename);
  if (!port) {
    return nullptr;
  }
  
  // Call the procedure with the port
  SCM_List proc_args;
  proc_args.data = port;
  proc_args.next = nullptr;
  proc_args.is_dotted = false;
  
  SCM *result;
  if (is_proc(proc)) {
    SCM_Procedure *proc_obj = cast<SCM_Procedure>(proc);
    result = apply_procedure(g_env.parent ? g_env.parent : &g_env, proc_obj, &proc_args);
  } else {
    SCM_Function *func_obj = cast<SCM_Function>(proc);
    SCM_List func_call;
    func_call.data = proc;
    func_call.next = &proc_args;
    result = eval_with_func(func_obj, &func_call);
  }
  
  // Close the port
  scm_c_close_output_port(port);
  
  return result;
}

// call-with-input-string: Create string port, call proc, then return result
SCM *scm_c_call_with_input_string(SCM_List *args) {
  if (!args || !args->data || !args->next || !args->next->data) {
    eval_error("call-with-input-string: requires 2 arguments (string proc)");
    return nullptr;
  }
  
  SCM *str = args->data;
  SCM *proc = args->next->data;
  
  if (!is_str(str)) {
    eval_error("call-with-input-string: first argument must be a string");
    return nullptr;
  }
  
  if (!is_proc(proc) && !is_func(proc)) {
    eval_error("call-with-input-string: second argument must be a procedure");
    return nullptr;
  }
  
  // Create the string port
  SCM *port = scm_c_open_input_string(str);
  if (!port) {
    return nullptr;
  }
  
  // Call the procedure with the port
  SCM_List proc_args;
  proc_args.data = port;
  proc_args.next = nullptr;
  proc_args.is_dotted = false;
  
  SCM *result;
  if (is_proc(proc)) {
    SCM_Procedure *proc_obj = cast<SCM_Procedure>(proc);
    result = apply_procedure(g_env.parent ? g_env.parent : &g_env, proc_obj, &proc_args);
  } else {
    SCM_Function *func_obj = cast<SCM_Function>(proc);
    SCM_List func_call;
    func_call.data = proc;
    func_call.next = &proc_args;
    result = eval_with_func(func_obj, &func_call);
  }
  
  // Port is automatically cleaned up (no explicit close needed for string ports)
  
  return result;
}

// call-with-output-string: Create string port, call proc, then return the string
SCM *scm_c_call_with_output_string(SCM_List *args) {
  if (!args || !args->data) {
    eval_error("call-with-output-string: requires 1 argument (proc)");
    return nullptr;
  }
  
  SCM *proc = args->data;
  
  if (!is_proc(proc) && !is_func(proc)) {
    eval_error("call-with-output-string: argument must be a procedure");
    return nullptr;
  }
  
  // Create the string port
  SCM *port = scm_c_open_output_string();
  if (!port) {
    return nullptr;
  }
  
  // Call the procedure with the port
  SCM_List proc_args;
  proc_args.data = port;
  proc_args.next = nullptr;
  proc_args.is_dotted = false;
  
  SCM *result;
  if (is_proc(proc)) {
    SCM_Procedure *proc_obj = cast<SCM_Procedure>(proc);
    result = apply_procedure(g_env.parent ? g_env.parent : &g_env, proc_obj, &proc_args);
  } else {
    SCM_Function *func_obj = cast<SCM_Function>(proc);
    SCM_List func_call;
    func_call.data = proc;
    func_call.next = &proc_args;
    result = eval_with_func(func_obj, &func_call);
  }
  
  // Get the output string
  SCM *output_str = scm_c_get_output_string(port);
  
  return output_str;
}

void init_port() {
  // Initialize EOF object
  scm_eof_object();
  
  // File ports
  scm_define_function("open-input-file", 1, 0, 0, scm_c_open_input_file);
  scm_define_function("open-output-file", 1, 0, 0, scm_c_open_output_file);
  scm_define_function("close-input-port", 1, 0, 0, scm_c_close_input_port);
  scm_define_function("close-output-port", 1, 0, 0, scm_c_close_output_port);
  
  // String ports
  scm_define_function("open-input-string", 1, 0, 0, scm_c_open_input_string);
  scm_define_function("open-output-string", 0, 0, 0, scm_c_open_output_string);
  scm_define_function("get-output-string", 1, 0, 0, scm_c_get_output_string);
  
  // Port operations
  scm_define_vararg_function("read", scm_c_read);
  scm_define_vararg_function("read-char", scm_c_read_char);
  scm_define_vararg_function("peek-char", scm_c_peek_char);
  scm_define_function("eof-object?", 1, 0, 0, scm_c_is_eof_object);
  scm_define_vararg_function("char-ready?", scm_c_char_ready);
  
  // Port predicates
  scm_define_function("input-port?", 1, 0, 0, scm_c_is_input_port);
  scm_define_function("output-port?", 1, 0, 0, scm_c_is_output_port);
  
  // Port write operations
  scm_define_vararg_function("write-char", scm_c_write_char);
  
  // Port wrappers
  scm_define_vararg_function("call-with-input-file", scm_c_call_with_input_file);
  scm_define_vararg_function("call-with-output-file", scm_c_call_with_output_file);
  scm_define_vararg_function("call-with-input-string", scm_c_call_with_input_string);
  scm_define_vararg_function("call-with-output-string", scm_c_call_with_output_string);
}

