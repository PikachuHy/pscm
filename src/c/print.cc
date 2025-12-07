#include "pscm.h"
void print_ast(SCM *ast) {
  if (is_proc(ast)) {
    auto proc = cast<SCM_Procedure>(ast);
    printf("#");
    printf("<");
    if (proc->name) {
      printf("procedure %s ", proc->name->data);
    }
    else {
      printf("procedure #f ");
    }
    print_list(proc->args);
    printf(">");
    return;
  }
  if (is_func(ast)) {
    auto func = cast<SCM_Function>(ast);
    printf("#");
    if (func->generic) {
      printf("<primitive-generic ");
    }
    else {
      printf("<builtin-func ");
    }
    assert(func->name);
    printf("%s", func->name->data);
    printf(">");
    return;
  }
  if (is_cont(ast)) {
    auto cont = cast<SCM_Continuation>(ast);
    printf("<");
    printf("continuation@%p", cont);
    printf(">");
    return;
  }
  if (is_num(ast)) {
    int64_t num = (int64_t)ast->value;
    printf("%lld", num);
    return;
  }
  if (is_sym(ast)) {
    auto sym = cast<SCM_Symbol>(ast);
    printf("%s", sym->data);
    return;
  }
  if (is_str(ast)) {
    auto str = cast<SCM_String>(ast);
    printf("\"%s\"", str->data);
    return;
  }
  if (is_pair(ast)) {
    auto l = cast<SCM_List>(ast);
    print_list(l);
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
    if (ast->value) {
      printf("#t");
    }
    else {
      printf("#f");
    }
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
  printf("%s:%d not supported %d\n", __FILE__, __LINE__, ast->type);
  exit(1);
}

// Helper function to print a dotted pair: (car . cdr)
static void _print_dotted_pair(SCM *car_val, SCM *cdr_val) {
  printf("(");
  print_ast(car_val);
  printf(" . ");
  print_ast(cdr_val);
  printf(")");
}

// Check if a 2-element structure should be printed as a dotted pair
static bool _should_print_as_pair(SCM_List *l, bool nested) {
  if (!l || !l->next || l->next->next) {
    return false; // Not a 2-element structure
  }
  
  SCM *cdr_val = l->next->data;
  SCM *car_val = l->data;
  
  // A dotted pair has atomic cdr (not pair, not nil)
  if (is_pair(cdr_val) || is_nil(cdr_val)) {
    return false;
  }
  
  // Print as pair if nested OR if car is not a number
  // (numbers in map results suggest it's a list, not a pair)
  return nested || !is_num(car_val);
}

static void _print_list(SCM_List *l, bool nested) {
  if (!l) {
    printf("()");
    return;
  }
  
  // Handle quote special form
  if (is_sym(l->data)) {
    auto sym = cast<SCM_Symbol>(l->data);
    if (strcmp(sym->data, "quote") == 0) {
      printf("'");
      if (l->next) {
        print_ast(l->next->data);
        assert(!l->next->next);
      } else {
        printf("()");
      }
      return;
    }
  }
  
  // Check if this should be printed as a dotted pair
  if (_should_print_as_pair(l, nested)) {
    _print_dotted_pair(l->data, l->next->data);
    return;
  }
  
  // Regular list printing
  printf("(");
  for (SCM_List *current = l; current; current = current->next) {
    if (current != l) {
      printf(" ");
    }
    // When printing elements of a list, pass nested=true for nested pairs
    if (is_pair(current->data)) {
      _print_list(cast<SCM_List>(current->data), true);
    } else {
      print_ast(current->data);
    }
  }
  printf(")");
}

void print_list(SCM_List *l) {
  _print_list(l, false);
}

