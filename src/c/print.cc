#include "pscm.h"

// Print context to track if we're inside a quasiquote expression
struct PrintContext {
  bool in_quasiquote;
  
  PrintContext() : in_quasiquote(false) {}
  PrintContext(bool in_qq) : in_quasiquote(in_qq) {}
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
    // Use %g for compact representation, but ensure decimal point is shown
    if (val == (double)(int64_t)val) {
      // Integer value stored as float: print with .0 to show it's a float
      printf("%.1f", val);
    } else {
      printf("%g", val);
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
      // Write format: with quotes
      printf("\"%s\"", str->data);
    } else {
      // Display format: without quotes
      printf("%s", str->data);
    }
    return;
  }
  if (is_pair(ast)) {
    auto l = cast<SCM_List>(ast);
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
  printf("%s:%d not supported %d\n", __FILE__, __LINE__, ast->type);
  exit(1);
}

// Helper function to print a dotted pair: (car . cdr)
static void _print_dotted_pair(SCM *car_val, SCM *cdr_val, const PrintContext &ctx) {
  printf("(");
  _print_ast_with_context(car_val, true, ctx);
  printf(" . ");
  _print_ast_with_context(cdr_val, true, ctx);
  printf(")");
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

static void _print_list(SCM_List *l, bool nested, const PrintContext &ctx) {
  if (!l) {
    printf("()");
    return;
  }
  
  // Check if this list starts with quasiquote - if so, set context
  PrintContext new_ctx = ctx;
  if (is_sym(l->data)) {
    auto sym = cast<SCM_Symbol>(l->data);
    if (strcmp(sym->data, "quasiquote") == 0) {
      new_ctx.in_quasiquote = true;
    }
  }
  
  // Handle quote special form
  // In quasiquote context, always use '... format for nested quotes
  // Outside quasiquote context, use (quote ...) for nested quotes
  if (is_sym(l->data)) {
    auto sym = cast<SCM_Symbol>(l->data);
    if (strcmp(sym->data, "quote") == 0) {
      // Check if the quoted expression is a list starting with unquote/quasiquote/unquote-splicing
      bool should_use_quote_syntax = false;
      if (l->next && is_pair(l->next->data)) {
        SCM_List *quoted_list = cast<SCM_List>(l->next->data);
        if (quoted_list->data && is_sym(quoted_list->data)) {
          SCM_Symbol *quoted_sym = cast<SCM_Symbol>(quoted_list->data);
          if (strcmp(quoted_sym->data, "unquote") == 0 ||
              strcmp(quoted_sym->data, "quasiquote") == 0 ||
              strcmp(quoted_sym->data, "unquote-splicing") == 0) {
            should_use_quote_syntax = true;
          }
        }
      }
      
      // Use '... format if:
      // 1. Not nested (top-level)
      // 2. Contains unquote/quasiquote/unquote-splicing
      // 3. In quasiquote context (even when nested) - ALL quotes use '... format
      // Outside quasiquote context, nested quotes always use (quote ...) format
      
      if (!nested || should_use_quote_syntax || ctx.in_quasiquote) {
        // Top-level, or contains special forms, or in quasiquote context
        printf("'");
        if (l->next) {
          _print_ast_with_context(l->next->data, true, new_ctx);
          assert(!l->next->next);
        } else {
          printf("()");
        }
        return;
      } else {
        // Nested quote outside quasiquote: print as (quote ...)
        printf("(quote");
        if (l->next) {
          printf(" ");
          _print_ast_with_context(l->next->data, true, new_ctx);
          assert(!l->next->next);
        }
        printf(")");
        return;
      }
    }
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
