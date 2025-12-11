#include "pscm.h"
#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Simple Scheme parser implementation from scratch

// Parser state
struct Parser {
  const char *input;
  const char *pos;
  const char *filename;
  int line;
  int column;
};

// Error reporting
static void parse_error(Parser *p, const char *msg) {
  fprintf(stderr, "%s:%d:%d: parse error: %s\n", 
          p->filename ? p->filename : "<input>", p->line, p->column, msg);
  exit(1);
}

// Skip whitespace and comments
static void skip_whitespace(Parser *p) {
  while (true) {
    while (isspace((unsigned char)*p->pos)) {
      if (*p->pos == '\n') {
        p->line++;
        p->column = 1;
      } else {
        p->column++;
      }
      p->pos++;
    }
    if (*p->pos == ';') {
      // Skip comment until end of line
      while (*p->pos && *p->pos != '\n') {
        p->pos++;
        p->column++;
      }
    } else {
      break;
    }
  }
}

// Check if we're at end of input
static bool is_eof(Parser *p) {
  skip_whitespace(p);
  return *p->pos == '\0';
}

// Peek at current character
static char peek(Parser *p) {
  skip_whitespace(p);
  return *p->pos;
}

// Consume current character
static char consume(Parser *p) {
  skip_whitespace(p);
  char c = *p->pos;
  if (c) {
    p->pos++;
    p->column++;
    if (c == '\n') {
      p->line++;
      p->column = 1;
    }
  }
  return c;
}

// Parse a number (only if it starts with a digit, or +/- followed by a digit)
static SCM *parse_number(Parser *p) {
  const char *start = p->pos;
  int start_line = p->line;
  int start_column = p->column;
  bool negative = false;
  
  // Only parse as number if it starts with a digit, or +/- followed by a digit
  if (*p->pos == '-') {
    // Check if next char is a digit
    if (!isdigit((unsigned char)p->pos[1])) {
      return nullptr; // Not a number, let it be parsed as symbol
    }
    negative = true;
    p->pos++;
    p->column++;
  } else if (*p->pos == '+') {
    // Check if next char is a digit
    if (!isdigit((unsigned char)p->pos[1])) {
      return nullptr; // Not a number, let it be parsed as symbol
    }
    p->pos++;
    p->column++;
  } else if (!isdigit((unsigned char)*p->pos)) {
    return nullptr;
  }
  
  // Parse integer part
  long int_part = 0;
  while (isdigit((unsigned char)*p->pos)) {
    int_part = int_part * 10 + (*p->pos - '0');
    p->pos++;
    p->column++;
  }
  
  // Check for decimal point
  bool is_float = false;
  double float_value = (double)int_part;
  
  if (*p->pos == '.') {
    is_float = true;
    p->pos++;
    p->column++;
  
    // Parse fractional part
    double fractional = 0.0;
    double divisor = 1.0;
    while (isdigit((unsigned char)*p->pos)) {
      fractional = fractional * 10 + (*p->pos - '0');
      divisor *= 10;
      p->pos++;
      p->column++;
    }
    float_value += fractional / divisor;
  }
  
  // Check for exponent (scientific notation)
  if (*p->pos == 'e' || *p->pos == 'E') {
    is_float = true;
    p->pos++;
    p->column++;
    
    // Parse exponent sign
    bool exp_negative = false;
    if (*p->pos == '-') {
      exp_negative = true;
      p->pos++;
      p->column++;
    } else if (*p->pos == '+') {
      p->pos++;
      p->column++;
    }
    
    // Parse exponent value
    long exp = 0;
    if (!isdigit((unsigned char)*p->pos)) {
      parse_error(p, "expected digit after exponent marker");
    }
    while (isdigit((unsigned char)*p->pos)) {
      exp = exp * 10 + (*p->pos - '0');
      p->pos++;
      p->column++;
    }
    
    // Apply exponent
    double multiplier = 1.0;
    for (long i = 0; i < exp; i++) {
      multiplier *= 10.0;
    }
    if (exp_negative) {
      float_value /= multiplier;
    } else {
      float_value *= multiplier;
    }
  }
  
  // Apply sign
  if (negative) {
    if (is_float) {
      float_value = -float_value;
    } else {
      int_part = -int_part;
    }
  }
  
  // Create SCM object
  SCM *scm = new SCM();
  if (is_float) {
    scm->type = SCM::FLOAT;
    scm->value = double_to_ptr(float_value);
  } else {
    scm->type = SCM::NUM;
    scm->value = (void *)int_part;
  }
  scm->source_loc = nullptr;
  set_source_location(scm, p->filename, start_line, start_column);
  return scm;
}
  
// Parse a character literal (#\A, #\., #\space, etc.)
static SCM *parse_char(Parser *p) {
  // Must match '#\'
  if (p->pos[0] != '#' || p->pos[1] != '\\') {
    return nullptr;
  }
  
  const char *start = p->pos;
  int start_line = p->line;
  int start_column = p->column;
  
  p->pos += 2;  // Skip #\
  p->column += 2;
  
  if (*p->pos == '\0') {
    parse_error(p, "unexpected end of input in character literal");
  }
  
  char ch;
  
  // Check for named characters
  if (isalpha((unsigned char)*p->pos)) {
    const char *name_start = p->pos;
    while (isalpha((unsigned char)*p->pos)) {
      p->pos++;
      p->column++;
    }
    
    int len = p->pos - name_start;
    // Check for common named characters
    if (len == 5 && strncmp(name_start, "space", 5) == 0) {
      ch = ' ';
    } else if (len == 7 && strncmp(name_start, "newline", 7) == 0) {
      ch = '\n';
    } else if (len == 3 && strncmp(name_start, "tab", 3) == 0) {
      ch = '\t';
    } else {
      // If not a named character, treat the first character as the character value
      // This handles cases like #\A where A is a letter
      p->pos = name_start;
      p->column = start_column + 2;
      ch = *p->pos;
      p->pos++;
      p->column++;
    }
  } else {
    // Single character (non-letter or any single char after #\)
    ch = *p->pos;
    p->pos++;
    p->column++;
  }
  
  SCM *scm = scm_from_char(ch);
  set_source_location(scm, p->filename, start_line, start_column);
  return scm;
}

// Forward declarations
static SCM *parse_expr(Parser *p);
static SCM *parse_unquote_in_list(Parser *p);

// Parse a vector (#(element1 element2 ...))
static SCM *parse_vector(Parser *p) {
  // Must match '#('
  if (p->pos[0] != '#' || p->pos[1] != '(') {
    return nullptr;
  }
  
  const char *start = p->pos;
  int start_line = p->line;
  int start_column = p->column;
  
  p->pos += 2;  // Skip #(
  p->column += 2;
  
  skip_whitespace(p);
  
  // Check for empty vector
  if (*p->pos == ')') {
    p->pos++; // consume ')'
    p->column++;
    
    auto vec = new SCM_Vector();
    vec->elements = nullptr;
    vec->length = 0;
    
    SCM *scm = wrap(vec);
    set_source_location(scm, p->filename, start_line, start_column);
    return scm;
  }
  
  // Parse vector elements
  SCM_List dummy = make_list_dummy();
  SCM_List *tail = &dummy;
  
  while (*p->pos && *p->pos != ')') {
    // Check for unquote/unquote-splicing in vector context
    SCM *elem = parse_unquote_in_list(p);
    if (!elem) {
      elem = parse_expr(p);
    }
    if (!elem) {
      parse_error(p, "expected expression in vector");
    }
    
    SCM_List *node = make_list(elem);
    tail->next = node;
    tail = node;
    
    skip_whitespace(p);
  }
  
  if (*p->pos != ')') {
    parse_error(p, "expected ')' to close vector");
  }
  p->pos++; // consume ')'
  p->column++;
  
  // Convert list to vector
  int count = 0;
  SCM_List *current = dummy.next;
  while (current) {
    count++;
    current = current->next;
  }
  
  auto vec = new SCM_Vector();
  vec->length = count;
  if (count > 0) {
    vec->elements = new SCM*[count];
    current = dummy.next;
    for (int i = 0; i < count; i++) {
      vec->elements[i] = current->data;
      current = current->next;
    }
  } else {
    vec->elements = nullptr;
  }
  
  SCM *scm = wrap(vec);
  set_source_location(scm, p->filename, start_line, start_column);
  return scm;
}

// Parse a string
static SCM *parse_string(Parser *p) {
  if (*p->pos != '"') {
    return nullptr;
  }
  int start_line = p->line;
  int start_column = p->column;
  p->pos++; // consume opening quote
  p->column++;
  
  const char *start = p->pos;
  int len = 0;
  bool escape = false;
  
  while (*p->pos) {
    if (escape) {
      escape = false;
      len++;
      p->pos++;
      p->column++;
      continue;
    }
    if (*p->pos == '\\') {
      escape = true;
      p->pos++;
      p->column++;
      continue;
    }
    if (*p->pos == '"') {
      break;
    }
    if (*p->pos == '\n') {
      p->line++;
      p->column = 1;
    }
    len++;
    p->pos++;
    p->column++;
  }
  
  if (*p->pos != '"') {
    parse_error(p, "unterminated string");
  }
  p->pos++; // consume closing quote
  p->column++;
  
  // Create string symbol
  char *str_data = new char[len + 1];
  const char *src = start;
  char *dst = str_data;
  escape = false;
  
  for (int i = 0; i < len; i++) {
    if (escape) {
      escape = false;
      switch (*src) {
        case 'n': *dst++ = '\n'; break;
        case 't': *dst++ = '\t'; break;
        case 'r': *dst++ = '\r'; break;
        case '\\': *dst++ = '\\'; break;
        case '"': *dst++ = '"'; break;
        default: *dst++ = *src; break;
      }
      src++;
    } else if (*src == '\\') {
      escape = true;
      src++;
    } else {
      *dst++ = *src++;
    }
  }
  *dst = '\0';
  
  SCM *scm = create_sym(str_data, len);
  scm->type = SCM::STR;
  set_source_location(scm, p->filename, start_line, start_column);
  delete[] str_data;
  return scm;
}

// Parse a symbol or special literal
static SCM *parse_symbol(Parser *p) {
  const char *start = p->pos;
  int start_line = p->line;
  int start_column = p->column;
  int len = 0;
  
  // Check for special literals first
  if (strncmp(p->pos, "#t", 2) == 0 && 
      (p->pos[2] == '\0' || isspace((unsigned char)p->pos[2]) || 
       p->pos[2] == ')' || p->pos[2] == '(' || p->pos[2] == ';')) {
    p->pos += 2;
    p->column += 2;
    SCM *result = scm_bool_true();
    set_source_location(result, p->filename, start_line, start_column);
    return result;
  }
  
  if (strncmp(p->pos, "#f", 2) == 0 && 
      (p->pos[2] == '\0' || isspace((unsigned char)p->pos[2]) || 
       p->pos[2] == ')' || p->pos[2] == '(' || p->pos[2] == ';')) {
    p->pos += 2;
    p->column += 2;
    SCM *result = scm_bool_false();
    set_source_location(result, p->filename, start_line, start_column);
    return result;
  }
  
  // Parse regular symbol
  // Symbols can contain: letters, digits, and special characters
  while (*p->pos && 
         (isalnum((unsigned char)*p->pos) || 
          strchr("!$%&*+-./:<=>?@^_~", *p->pos) != nullptr)) {
    len++;
    p->pos++;
    p->column++;
  }
  
  if (len == 0) {
    return nullptr;
  }
  
  char *sym_data = new char[len + 1];
  memcpy(sym_data, start, len);
  sym_data[len] = '\0';
  
  SCM *scm = create_sym(sym_data, len);
  set_source_location(scm, p->filename, start_line, start_column);
  delete[] sym_data;
  return scm;
}


// Forward declarations
static SCM *parse_expr(Parser *p);
static SCM *parse_unquote_in_list(Parser *p);

// Parse a quoted expression
static SCM *parse_quote(Parser *p) {
  if (*p->pos != '\'') {
    return nullptr;
  }
  p->pos++; // consume quote
  p->column++;
  
  // After quote, we need to parse the expression
  // But in list context (like in quasiquote), we might have ',name
  // which should be parsed as (quote ,name) where ,name is an unquote
  // So we need to check for unquote first
  SCM *quoted = parse_unquote_in_list(p);
  if (!quoted) {
    quoted = parse_expr(p);
  }
  if (!quoted) {
    parse_error(p, "expected expression after quote");
  }
  
  // Build (quote expr)
  SCM *quote_sym = scm_sym_quote();
  SCM_List *list = make_list(quote_sym, quoted);
  
  SCM *scm = new SCM();
  scm->type = SCM::LIST;
  scm->value = list;
  return scm;
}

// Parse a quasiquoted expression (backquote `)
static SCM *parse_quasiquote(Parser *p) {
  if (*p->pos != '`') {
    return nullptr;
  }
  p->pos++; // consume backquote
  p->column++;
  
  // After quasiquote, we might have unquote directly (like `,expr)
  // Check for unquote first
  SCM *quoted = parse_unquote_in_list(p);
  if (!quoted) {
    quoted = parse_expr(p);
  }
  if (!quoted) {
    parse_error(p, "expected expression after quasiquote");
  }
  
  // Build (quasiquote expr)
  SCM *quasiquote_sym = scm_sym_quasiquote();
  SCM_List *list = make_list(quasiquote_sym, quoted);
  
  SCM *scm = new SCM();
  scm->type = SCM::LIST;
  scm->value = list;
  return scm;
}

// Parse an unquote expression (comma ,)
// This is called when we encounter a comma in a list context
static SCM *parse_unquote_in_list(Parser *p) {
  if (*p->pos != ',') {
    return nullptr;
  }
  
  // Check for unquote-splicing (,@)
  if (p->pos[1] == '@') {
    p->pos += 2; // consume ,@
    p->column += 2;
    
    SCM *expr = parse_expr(p);
    if (!expr) {
      parse_error(p, "expected expression after unquote-splicing");
    }
    
    // Build (unquote-splicing expr)
    SCM *unquote_splicing_sym = scm_sym_unquote_splicing();
    SCM_List *list = make_list(unquote_splicing_sym, expr);
    
    SCM *scm = new SCM();
    scm->type = SCM::LIST;
    scm->value = list;
    return scm;
  }
  
  // Regular unquote
  p->pos++; // consume comma
  p->column++;
  
  SCM *expr = parse_expr(p);
  if (!expr) {
    parse_error(p, "expected expression after unquote");
  }
  
  // Build (unquote expr)
  SCM *unquote_sym = scm_sym_unquote();
  SCM_List *list = make_list(unquote_sym, expr);
  
  SCM *scm = new SCM();
  scm->type = SCM::LIST;
  scm->value = list;
  return scm;
}

// Parse a list
static SCM *parse_list(Parser *p) {
  if (*p->pos != '(') {
    return nullptr;
  }
  int start_line = p->line;
  int start_column = p->column;
  p->pos++; // consume '('
  p->column++;
  
  SCM_List dummy = make_list_dummy();
  SCM_List *tail = &dummy;
  
  skip_whitespace(p);
  
  // Check for empty list
  if (*p->pos == ')') {
    p->pos++; // consume ')'
    p->column++;
    SCM *result = scm_nil();
    set_source_location(result, p->filename, start_line, start_column);
    return result;
  }
  
  // Parse list elements
  while (*p->pos && *p->pos != ')') {
    // Check for quote first (since ',name should be parsed as (quote ,name))
    SCM *elem = parse_quote(p);
    if (!elem) {
      // Check for unquote/unquote-splicing in list context
      elem = parse_unquote_in_list(p);
      if (!elem) {
        elem = parse_expr(p);
      }
    }
    if (!elem) {
      parse_error(p, "expected expression in list");
    }
    
    SCM_List *node = make_list(elem);
    tail->next = node;
    tail = node;
    
    skip_whitespace(p);
    
    // Check for dotted pair
    // But first, check if the dot is part of a symbol (like ... or !..)
    // A dot is a dotted pair marker only if it's followed by whitespace and then an expression
    if (*p->pos == '.') {
      // Peek ahead to see if this is part of a symbol
      const char *peek_pos = p->pos + 1;
      
      // Check if dot is immediately followed by another dot or symbol character
      // If so, it's part of a symbol (like ...)
      if (*peek_pos == '.' || 
          isalnum((unsigned char)*peek_pos) || 
          strchr("!$%&*+-./:<=>?@^_~", *peek_pos) != nullptr) {
        // This dot is part of a symbol, continue parsing the next element
        // The dot will be consumed as part of the symbol
        continue;
      }
      
      // Skip whitespace to check what comes after the dot
      while (isspace((unsigned char)*peek_pos)) {
        peek_pos++;
      }
      
      // If dot is followed by ')' or end of input, it's a dotted pair marker
      if (*peek_pos == ')' || *peek_pos == '\0') {
        // This is a dotted pair marker
        p->pos++;
        p->column++;
        skip_whitespace(p);
        
        // Check for unquote/unquote-splicing in dotted pair context
        SCM *cdr = parse_unquote_in_list(p);
        if (!cdr) {
          cdr = parse_expr(p);
        }
        if (!cdr) {
          parse_error(p, "expected expression after dot");
        }
        
        // Create node for cdr and mark it as dotted pair
        if (is_pair(cdr)) {
          // When cdr is a pair, wrap it in a node to store it
          // This allows us to mark the wrapper as dotted, while keeping the
          // cdr list itself unmarked (if it's a proper list)
          tail->next = make_list(cdr);
          tail->next->is_dotted = true;  // Mark the wrapper node as dotted
          // The cdr list itself (cdr) is not modified - its internal structure
          // remains unchanged. The printing code will check if cdr is a proper
          // list and expand it: (a b . (c d)) -> (a b c d)
        } else {
          tail->next = make_list(cdr);
          tail->next->is_dotted = true;  // Mark as dotted pair's cdr
        }
        skip_whitespace(p);
        break;
      }
      
      // Otherwise, the dot is followed by something else (might be part of symbol or dotted pair)
      // Try to parse as dotted pair first
      p->pos++;
      p->column++;
      skip_whitespace(p);
      
      // Check for unquote/unquote-splicing in dotted pair context
      SCM *cdr = parse_unquote_in_list(p);
      if (!cdr) {
        cdr = parse_expr(p);
      }
      if (!cdr) {
        parse_error(p, "expected expression after dot");
      }
      
      // Create node for cdr and mark it as dotted pair
      if (is_pair(cdr)) {
        tail->next = make_list(cdr);
        tail->next->is_dotted = true;
      } else {
        tail->next = make_list(cdr);
        tail->next->is_dotted = true;
      }
      skip_whitespace(p);
      break;
    }
  }
  
  if (*p->pos != ')') {
    parse_error(p, "expected ')' to close list");
  }
  p->pos++; // consume ')'
  p->column++;
  
  if (dummy.next) {
    SCM *scm = new SCM();
    scm->type = SCM::LIST;
    scm->value = dummy.next;
    scm->source_loc = nullptr;  // Initialize to nullptr
    set_source_location(scm, p->filename, start_line, start_column);
    return scm;
  }
  SCM *result = scm_nil();
  set_source_location(result, p->filename, start_line, start_column);
  return result;
}

// Parse an expression
static SCM *parse_expr(Parser *p) {
  skip_whitespace(p);
  
  if (*p->pos == '\0') {
    return nullptr;
  }
  
  // Try different parsers
  SCM *result = nullptr;
  
  // Quote
  if (*p->pos == '\'') {
    return parse_quote(p);
  }
  
  // Quasiquote (backquote)
  if (*p->pos == '`') {
    return parse_quasiquote(p);
  }
  
  // List
  if (*p->pos == '(') {
    return parse_list(p);
  }
  
  // String
  if (*p->pos == '"') {
    return parse_string(p);
  }
  
  // Character literal (#\A, #\., etc.)
  result = parse_char(p);
  if (result) {
    return result;
  }
  
  // Vector literal (#(element1 element2 ...))
  result = parse_vector(p);
  if (result) {
    return result;
  }
  
  // Special case: check for 1+ and 1- symbols before parsing as number
  // These are valid Scheme identifiers that start with a digit
  if ((strncmp(p->pos, "1+", 2) == 0 && 
       (p->pos[2] == '\0' || isspace((unsigned char)p->pos[2]) || 
        p->pos[2] == ')' || p->pos[2] == '(' || p->pos[2] == ';')) ||
      (strncmp(p->pos, "1-", 2) == 0 && 
       (p->pos[2] == '\0' || isspace((unsigned char)p->pos[2]) || 
        p->pos[2] == ')' || p->pos[2] == '(' || p->pos[2] == ';'))) {
    // Parse as symbol instead of number
    result = parse_symbol(p);
    if (result) {
      return result;
    }
  }
  
  // Number
  result = parse_number(p);
  if (result) {
    return result;
  }
  
  // Symbol or special literal
  result = parse_symbol(p);
  if (result) {
    return result;
  }
  
  // Unknown token
  parse_error(p, "unexpected character");
  return nullptr;
}

// Parse a single expression from input
SCM *parse(const char *s) {
  Parser p;
  p.input = s;
  p.pos = s;
  p.filename = nullptr;
  p.line = 1;
  p.column = 1;
  
  SCM *result = parse_expr(&p);
  if (!result) {
    parse_error(&p, "expected expression");
  }
  
  skip_whitespace(&p);
  if (*p.pos != '\0') {
    parse_error(&p, "extra input after expression");
  }
  
  return result;
}

// Read file content
static char *read_file_content(const char *filename) {
  FILE *f = fopen(filename, "r");
  if (!f) {
    fprintf(stderr, "ERROR: cannot open file: %s\n", filename);
    return nullptr;
  }
  
  // Get file size
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  
  // Allocate buffer
  char *buffer = new char[size + 1];
  size_t read = fread(buffer, 1, size, f);
  buffer[read] = '\0';
  fclose(f);
  
  return buffer;
}

// Parse file and return list of expressions
SCM_List *parse_file(const char *filename) {
  char *content = read_file_content(filename);
  if (!content) {
    return nullptr;
  }
  
  Parser p;
  p.input = content;
  p.pos = content;
  p.filename = filename;
  p.line = 1;
  p.column = 1;
  
  SCM_List dummy = make_list_dummy();
  SCM_List *tail = &dummy;
  
  while (!is_eof(&p)) {
    SCM *expr = parse_expr(&p);
    if (!expr) {
      break;
    }
    
    SCM_List *node = make_list(expr);
    tail->next = node;
    tail = node;
    
    skip_whitespace(&p);
  }
  
  delete[] content;
  return dummy.next;
}
