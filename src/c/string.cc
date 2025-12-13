#include "pscm.h"
#include "eval.h"

// string-length: Return the number of characters in string
SCM *scm_c_string_length(SCM *str) {
  if (!is_str(str)) {
    eval_error("string-length: expected string");
    return nullptr;
  }
  SCM_String *s = cast<SCM_String>(str);
  SCM *scm = new SCM();
  scm->type = SCM::NUM;
  scm->value = (void*)(int64_t)s->len;
  scm->source_loc = nullptr;
  return scm;
}

// make-string: Create a new string of length k, optionally filled with char
// Helper function for implementation
static SCM *scm_c_make_string_impl(SCM *len_scm, SCM *char_scm) {
  if (!is_num(len_scm)) {
    eval_error("make-string: first argument must be an integer");
    return nullptr;
  }
  int64_t len = (int64_t)len_scm->value;
  if (len < 0) {
    eval_error("make-string: length must be non-negative");
    return nullptr;
  }
  
  // Get optional character (default to space)
  char fill_char = ' ';
  if (char_scm) {
    if (!is_char(char_scm)) {
      eval_error("make-string: second argument must be a character");
      return nullptr;
    }
    fill_char = scm_to_char(char_scm);
  }
  
  // Create string
  SCM_String *s = new SCM_String();
  s->data = new char[len + 1];
  for (int i = 0; i < len; i++) {
    s->data[i] = fill_char;
  }
  s->data[len] = '\0';
  s->len = len;
  
  SCM *scm = new SCM();
  scm->type = SCM::STR;
  scm->value = s;
  scm->source_loc = nullptr;
  return scm;
}

// Wrapper for variable arguments
SCM *scm_c_make_string(SCM_List *args) {
  if (!args || !args->data) {
    eval_error("make-string: requires at least 1 argument");
    return nullptr;
  }
  
  SCM *len_scm = args->data;
  SCM *char_scm = (args->next && args->next->data) ? args->next->data : nullptr;
  return scm_c_make_string_impl(len_scm, char_scm);
}

// string-ref: Return character at index k of string
SCM *scm_c_string_ref(SCM *str, SCM *k) {
  if (!is_str(str)) {
    eval_error("string-ref: expected string");
    return nullptr;
  }
  if (!is_num(k)) {
    eval_error("string-ref: expected integer index");
    return nullptr;
  }
  
  SCM_String *s = cast<SCM_String>(str);
  int64_t idx = (int64_t)k->value;
  
  if (idx < 0 || idx >= s->len) {
    eval_error("string-ref: index out of range");
    return nullptr;
  }
  
  return scm_from_char(s->data[idx]);
}

// string-set!: Store character chr at index k of string
SCM *scm_c_string_set(SCM *str, SCM *k, SCM *chr) {
  if (!is_str(str)) {
    eval_error("string-set!: expected string");
    return nullptr;
  }
  if (!is_num(k)) {
    eval_error("string-set!: expected integer index");
    return nullptr;
  }
  if (!is_char(chr)) {
    eval_error("string-set!: expected character");
    return nullptr;
  }
  
  SCM_String *s = cast<SCM_String>(str);
  int64_t idx = (int64_t)k->value;
  
  if (idx < 0 || idx >= s->len) {
    eval_error("string-set!: index out of range");
    return nullptr;
  }
  
  s->data[idx] = scm_to_char(chr);
  return scm_none(); // Return unspecified value
}

// display: Output the argument using display format (without quotes for strings)
SCM *scm_c_display(SCM *arg) {
  print_ast(arg, false); // Use display format (write_mode = false)
  return scm_none(); // Return unspecified value
}

// write: Output the argument using write format (with quotes for strings)
SCM *scm_c_write(SCM *arg) {
  print_ast(arg, true); // Use write format (write_mode = true)
  return scm_none(); // Return unspecified value
}

// newline: Output a newline character (optionally to a port)
SCM *scm_c_newline(SCM_List *args) {
  // args is the list of arguments (not including function name)
  // If no arguments, args is nullptr or args->data is nullptr
  if (!args) {
    // No arguments: output to stdout
    printf("\n");
  } else if (!args->data) {
    // Empty argument list: output to stdout
    printf("\n");
  } else {
    // One argument: port (not yet implemented, just output to stdout for now)
    printf("\n");
  }
  return scm_none();
}

// string=?: Compare two strings for equality (case-sensitive)
SCM *scm_c_string_eq(SCM *str1, SCM *str2) {
  if (!is_str(str1)) {
    eval_error("string=?: first argument must be a string");
    return nullptr;
  }
  if (!is_str(str2)) {
    eval_error("string=?: second argument must be a string");
    return nullptr;
  }
  
  auto s1 = cast<SCM_String>(str1);
  auto s2 = cast<SCM_String>(str2);
  
  if (s1->len != s2->len) {
    return scm_bool_false();
  }
  
  if (strncmp(s1->data, s2->data, s1->len) == 0) {
    return scm_bool_true();
  }
  return scm_bool_false();
}

// string: Create a string from characters
SCM *scm_c_string(SCM_List *args) {
  if (!args) {
    // No arguments: return empty string
    SCM_String *s = new SCM_String();
    s->data = new char[1];
    s->data[0] = '\0';
    s->len = 0;
    SCM *scm = new SCM();
    scm->type = SCM::STR;
    scm->value = s;
    scm->source_loc = nullptr;
    return scm;
  }
  
  // Count characters
  int len = 0;
  SCM_List *current = args;
  while (current) {
    if (!is_char(current->data)) {
      eval_error("string: all arguments must be characters");
      return nullptr;
    }
    len++;
    if (!current->next || is_nil(wrap(current->next))) {
      break;
    }
    current = current->next;
  }
  
  // Create string from characters
  SCM_String *s = new SCM_String();
  s->data = new char[len + 1];
  s->len = len;
  current = args;
  int i = 0;
  while (current && i < len) {
    s->data[i] = scm_to_char(current->data);
    i++;
    if (!current->next || is_nil(wrap(current->next))) {
      break;
    }
    current = current->next;
  }
  s->data[len] = '\0';
  
  SCM *scm = new SCM();
  scm->type = SCM::STR;
  scm->value = s;
  scm->source_loc = nullptr;
  return scm;
}

// string->number: Convert a string to a number
// (string->number string [radix]) -> number or #f
SCM *scm_c_string_to_number(SCM_List *args) {
  if (!args || !args->data) {
    eval_error("string->number: requires at least 1 argument");
    return nullptr;
  }
  
  SCM *str_scm = args->data;
  if (!is_str(str_scm)) {
    eval_error("string->number: expected string");
    return nullptr;
  }
  
  SCM_String *s = cast<SCM_String>(str_scm);
  
  // Check for optional radix argument
  int radix = 10;
  if (args->next && args->next->data) {
    SCM *radix_scm = args->next->data;
    if (!is_num(radix_scm)) {
      eval_error("string->number: radix must be a number");
      return nullptr;
    }
    radix = (int)(int64_t)radix_scm->value;
    if (radix < 2 || radix > 36) {
      eval_error("string->number: radix must be between 2 and 36");
      return nullptr;
    }
  }
  
  // Empty string returns #f
  if (s->len == 0) {
    return scm_bool_false();
  }
  
  // Create a null-terminated string for parsing
  char *str = new char[s->len + 1];
  memcpy(str, s->data, s->len);
  str[s->len] = '\0';
  
  // Try to parse as number
  // For now, we'll use a simple approach: try strtol/strtod
  // This doesn't handle all Scheme number formats, but covers basic cases
  char *endptr;
  
  // Check if it's a float (contains '.', 'e', or 'E')
  bool is_float_str = false;
  for (int i = 0; i < s->len; i++) {
    if (str[i] == '.' || str[i] == 'e' || str[i] == 'E') {
      is_float_str = true;
      break;
    }
  }
  
  SCM *result = nullptr;
  
  if (is_float_str && radix == 10) {
    // Parse as float
    double val = strtod(str, &endptr);
    if (endptr == str || *endptr != '\0') {
      // Failed to parse
      result = scm_bool_false();
    } else {
      result = scm_from_double(val);
    }
  } else {
    // Parse as integer
    long val = strtol(str, &endptr, radix);
    if (endptr == str || *endptr != '\0') {
      // Failed to parse
      result = scm_bool_false();
    } else {
      // Create number directly
      SCM *num = new SCM();
      num->type = SCM::NUM;
      num->value = (void *)(int64_t)val;
      num->source_loc = nullptr;
      result = num;
    }
  }
  
  delete[] str;
  return result;
}

void init_string() {
  scm_define_function("string-length", 1, 0, 0, scm_c_string_length);
  scm_define_vararg_function("make-string", scm_c_make_string);
  scm_define_function("string-ref", 2, 0, 0, scm_c_string_ref);
  scm_define_function("string-set!", 3, 0, 0, scm_c_string_set);
  scm_define_function("string=?", 2, 0, 0, scm_c_string_eq);
  scm_define_vararg_function("string", scm_c_string);
  scm_define_function("display", 1, 0, 0, scm_c_display);
  scm_define_function("write", 1, 0, 0, scm_c_write);
  scm_define_vararg_function("newline", scm_c_newline);
  scm_define_vararg_function("string->number", scm_c_string_to_number);
}

