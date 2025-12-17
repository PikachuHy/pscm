#include "pscm.h"

// Print context to track if we're inside a quasiquote expression
struct PrintContext {
  bool in_quasiquote;
  bool in_unquote;
  
  PrintContext() : in_quasiquote(false), in_unquote(false) {}
  PrintContext(bool in_qq, bool in_unq = false) : in_quasiquote(in_qq), in_unquote(in_unq) {}
};

// Forward declarations
static void _print_list(SCM_List *l, bool nested, const PrintContext &ctx);
static void _print_ast_with_context(SCM *ast, bool write_mode, const PrintContext &ctx);

void print_ast(SCM *ast, bool write_mode) {
  PrintContext ctx;
  _print_ast_with_context(ast, write_mode, ctx);
}

static void _print_ast_with_context(SCM *ast, bool write_mode, const PrintContext &ctx) {
  if (is_proc(ast)) {
    auto proc = cast<SCM_Procedure>(ast);
    printf("#<");
    if (proc->name) {
      printf("procedure %s ", proc->name->data);
    } else {
      printf("procedure #f ");
    }
    print_list(proc->args);
    printf(">");
    return;
  }
  if (is_func(ast)) {
    auto func = cast<SCM_Function>(ast);
    assert(func->name);
    printf("#<%s %s>", func->generic ? "primitive-generic" : "builtin-func", func->name->data);
    return;
  }
  if (is_cont(ast)) {
    auto cont = cast<SCM_Continuation>(ast);
    printf("#<continuation@%p>", cont);
    return;
  }
  if (is_num(ast)) {
    int64_t num = (int64_t)ast->value;
    printf("%lld", num);
    return;
  }
  if (is_float(ast)) {
    double val = ptr_to_double(ast->value);
    // Always show at least one decimal place for floats to preserve the float representation
    // Match Guile's precision: use %.14g for better precision (14 significant digits)
    if (val == (double)(int64_t)val) {
      // Integer value stored as float: print with .0 to show it's a float
      printf("%.1f", val);
    } else {
      // Use %.14g to match Guile's precision (14 significant digits)
      // This will show enough precision for sqrt(2) = 1.4142135623731
      printf("%.14g", val);
    }
    return;
  }
  if (is_ratio(ast)) {
    SCM_Rational *rat = cast<SCM_Rational>(ast);
    printf("%lld/%lld", rat->numerator, rat->denominator);
    return;
  }
  if (is_char(ast)) {
    char ch = ptr_to_char(ast->value);
    // write_mode: print as #\A, #\., etc. (write format)
    // !write_mode: print as A, ., etc. (display format)
    if (write_mode) {
      // Write format: #\A, #\., #\space, etc.
      // Follow guile convention: space -> #\space, newline -> #\newline, tab -> #\ht
      if (ch == ' ') {
        printf("#\\space");
      } else if (ch == '\n') {
        printf("#\\newline");
      } else if (ch == '\t') {
        printf("#\\ht");
      } else {
        printf("#\\%c", ch);
      }
    } else {
      // Display format: just the character
      printf("%c", ch);
    }
    return;
  }
  if (is_sym(ast)) {
    auto sym = cast<SCM_Symbol>(ast);
    // Keywords (symbols starting with ':') should be printed as #:keyword-name
    if (sym->data && sym->data[0] == ':') {
      printf("#%s", sym->data);
    } else {
      printf("%s", sym->data);
    }
    return;
  }
  if (is_str(ast)) {
    auto str = cast<SCM_String>(ast);
    if (write_mode) {
      // Write format: with quotes and escaped special characters
      printf("\"");
      for (size_t i = 0; i < str->len; i++) {
        char c = str->data[i];
        switch (c) {
          case '\\': printf("\\\\"); break;
          case '"': printf("\\\""); break;
          case '\n': printf("\\n"); break;
          case '\t': printf("\\t"); break;
          case '\r': printf("\\r"); break;
          default:
            if (c >= 32 && c < 127) {
              printf("%c", c);
            } else {
              // For non-printable characters, use octal escape
              printf("\\%03o", (unsigned char)c);
            }
            break;
        }
      }
      printf("\"");
    } else {
      // Display format: without quotes
      printf("%s", str->data);
    }
    return;
  }
  if (is_port(ast)) {
    auto port = cast<SCM_Port>(ast);
    printf("#<");
    if (port->is_input) {
      printf("input");
    } else {
      printf("output");
    }
    if (port->port_type == PORT_FILE_INPUT || port->port_type == PORT_FILE_OUTPUT) {
      printf(": file");
    } else if (port->port_type == PORT_STRING_INPUT || port->port_type == PORT_STRING_OUTPUT) {
      printf(": string");
    }
    printf(" port>");
    return;
  }
  if (is_pair(ast)) {
    auto l = cast<SCM_List>(ast);
    if (!l) {
      type_error(ast, "pair");
    }
    _print_list(l, false, ctx);
    return;
  }
  if (is_vector(ast)) {
    auto vec = cast<SCM_Vector>(ast);
    printf("#(");
    for (size_t i = 0; i < vec->length; i++) {
      if (i > 0) {
        printf(" ");
      }
      // Pass context to vector elements
      _print_ast_with_context(vec->elements[i], write_mode, ctx);
    }
    printf(")");
    return;
  }
  if (is_nil(ast)) {
    printf("()");
    return;
  }
  if (is_none(ast)) {
    printf("none");
    return;
  }
  if (is_bool(ast)) {
    printf("%s", ast->value ? "#t" : "#f");
    return;
  }
  if (is_macro(ast)) {
    auto macro = cast<SCM_Macro>(ast);
    printf("#<macro!");
    if (macro->name) {
      printf(" %s", macro->name->data);
    }
    printf(">");
    return;
  }
  if (is_hash_table(ast)) {
    auto hash_table = cast<SCM_HashTable>(ast);
    printf("#<hash-table %zu/%zu>", hash_table->size, hash_table->capacity);
    return;
  }
  if (is_promise(ast)) {
    auto promise = cast<SCM_Promise>(ast);
    printf("#<promise");
    if (promise->is_forced) {
      printf(" forced");
    } else {
      printf(" pending");
    }
    printf(">");
    return;
  }
  if (is_module(ast)) {
    auto module = cast<SCM_Module>(ast);
    printf("#<");
    // Print module kind (default to "module")
    if (module->kind) {
      printf("%s", module->kind->data);
    } else {
      printf("module");
    }
    // Print module name if available
    if (module->name) {
      printf(" ");
      _print_list(module->name, false, ctx);
    }
    // Print module address in hex
    printf(" %p", (void *)module);
    printf(">");
    return;
  }
  printf("%s:%d not supported %d\n", __FILE__, __LINE__, ast->type);
  exit(1);
}

// Helper function to check if a list is a dotted pair by finding the last node
// and checking its is_dotted flag
static bool _is_dotted_pair(SCM_List *l) {
  if (!l || !l->next) {
    return false;
  }
  
  // Find the last node
  SCM_List *last = l;
  while (last->next) {
    last = last->next;
  }
  
  // Check if the last node is marked as dotted pair's cdr
  return last->is_dotted;
}

// Helper function to check if a list starts with a specific symbol
static bool _list_starts_with(SCM_List *l, const char *sym_name) {
  if (!l || !l->data || !is_sym(l->data)) {
    return false;
  }
  SCM_Symbol *sym = cast<SCM_Symbol>(l->data);
  return strcmp(sym->data, sym_name) == 0;
}

static void _print_list(SCM_List *l, bool nested, const PrintContext &ctx) {
  if (!l) {
    printf("()");
    return;
  }
  
  PrintContext new_ctx = ctx;
  
  // Handle quasiquote special form
  if (_list_starts_with(l, "quasiquote")) {
    new_ctx.in_quasiquote = true;
    printf("(quasiquote");
    if (l->next) {
      printf(" ");
      _print_ast_with_context(l->next->data, true, new_ctx);
      assert(!l->next->next);
    } else {
      printf(" ()");
    }
    printf(")");
    return;
  }
  
  // Handle quote special form
  if (_list_starts_with(l, "quote")) {
    // In quasiquote context, use '... format for quotes, unless we're inside an unquote
    // This matches the test expectations:
    // - '(unquote name) should print as '(unquote name)
    // - (unquote (quote (unquote name))) should print as (unquote (quote (unquote name)))
    if (ctx.in_quasiquote && !ctx.in_unquote) {
      printf("'");
      if (l->next) {
        _print_ast_with_context(l->next->data, true, new_ctx);
        assert(!l->next->next);
      } else {
        printf("()");
      }
    } else {
      // Use (quote ...) format when not in quasiquote context or inside unquote
      printf("(quote");
      if (l->next) {
        printf(" ");
        _print_ast_with_context(l->next->data, true, new_ctx);
        assert(!l->next->next);
      }
      printf(")");
    }
    return;
  }
  
  // Handle unquote special form
  if (_list_starts_with(l, "unquote")) {
    new_ctx.in_unquote = true;
    printf("(unquote");
    if (l->next) {
      printf(" ");
      _print_ast_with_context(l->next->data, true, new_ctx);
      assert(!l->next->next);
    }
    printf(")");
    return;
  }
  
  // Handle unquote-splicing special form
  if (_list_starts_with(l, "unquote-splicing")) {
    printf("(unquote-splicing");
    if (l->next) {
      printf(" ");
      _print_ast_with_context(l->next->data, true, new_ctx);
      assert(!l->next->next);
    }
    printf(")");
    return;
  }
  
  // Check if this is a dotted pair using is_dotted flag
  bool is_dotted_pair = _is_dotted_pair(l);
  
  if (is_dotted_pair) {
    // Find the last node (which has is_dotted = true)
    SCM_List *prev = nullptr;
    SCM_List *last = l;
    while (last->next) {
      prev = last;
      last = last->next;
    }
    
    // Check if cdr is a proper list - if so, expand it
    SCM *cdr_val = last->data;
    if (is_pair(cdr_val)) {
      // cdr is a list - check if it's a proper list
      SCM_List *cdr_list = cast<SCM_List>(cdr_val);
      bool cdr_is_proper = !_is_dotted_pair(cdr_list);
      
      if (cdr_is_proper) {
        // Expand: (a b . (c d)) -> (a b c d)
        printf("(");
        SCM_List *current = l;
        while (current && current != last) {
          if (current != l) {
            printf(" ");
          }
          if (is_pair(current->data)) {
            _print_list(cast<SCM_List>(current->data), true, new_ctx);
          } else {
            _print_ast_with_context(current->data, true, new_ctx);
          }
          current = current->next;
        }
        // Now print the cdr list elements
        SCM_List *cdr_current = cdr_list;
        while (cdr_current) {
          if (l != last || cdr_current != cdr_list) {
            printf(" ");
          }
          if (is_pair(cdr_current->data)) {
            _print_list(cast<SCM_List>(cdr_current->data), true, new_ctx);
          } else {
            _print_ast_with_context(cdr_current->data, true, new_ctx);
          }
          cdr_current = cdr_current->next;
        }
        printf(")");
        return;
      }
    }
    
    // Print as dotted pair: (a b c . d)
    printf("(");
    SCM_List *current = l;
    while (current) {
      if (current != l) {
        printf(" ");
      }
      
      // If this is the last element before the dotted pair cdr
      if (current == prev) {
        _print_ast_with_context(current->data, true, new_ctx);
        printf(" . ");
        _print_ast_with_context(last->data, true, new_ctx);
        break;
      }
      
      // Print element
      if (is_pair(current->data)) {
        _print_list(cast<SCM_List>(current->data), true, new_ctx);
      } else {
        _print_ast_with_context(current->data, true, new_ctx);
      }
      
      current = current->next;
    }
    printf(")");
  } else {
    // Regular list printing: (a b c)
    printf("(");
    SCM_List *current = l;
    while (current) {
      if (current != l) {
        printf(" ");
      }
      
      // Print element
      if (is_pair(current->data)) {
        _print_list(cast<SCM_List>(current->data), true, new_ctx);
      } else {
        _print_ast_with_context(current->data, true, new_ctx);
      }
      
      current = current->next;
    }
    printf(")");
  }
}

void print_list(SCM_List *l) {
  PrintContext ctx;
  _print_list(l, false, ctx);
}
