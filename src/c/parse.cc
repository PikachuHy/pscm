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
  
  long value = 0;
  while (isdigit((unsigned char)*p->pos)) {
    value = value * 10 + (*p->pos - '0');
    p->pos++;
    p->column++;
  }
  
  if (negative) {
    value = -value;
  }
  
  SCM *scm = new SCM();
  scm->type = SCM::NUM;
  scm->value = (void *)value;
  scm->source_loc = nullptr;  // Initialize to nullptr
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


// Forward declaration
static SCM *parse_expr(Parser *p);

// Parse a quoted expression
static SCM *parse_quote(Parser *p) {
  if (*p->pos != '\'') {
    return nullptr;
  }
  p->pos++; // consume quote
  p->column++;
  
  SCM *quoted = parse_expr(p);
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
  
  SCM *quoted = parse_expr(p);
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
    // Check for unquote/unquote-splicing in list context
    SCM *elem = parse_unquote_in_list(p);
    if (!elem) {
      elem = parse_expr(p);
    }
    if (!elem) {
      parse_error(p, "expected expression in list");
    }
    
    SCM_List *node = make_list(elem);
    tail->next = node;
    tail = node;
    
    skip_whitespace(p);
    
    // Check for dotted pair
    if (*p->pos == '.') {
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
      
      tail->next = is_pair(cdr) ? cast<SCM_List>(cdr) : make_list(cdr);
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
