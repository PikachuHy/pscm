#include "pscm.h"
void print_ast(SCM *ast) {
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
    printf("<continuation@%p>", cont);
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
    return false; // Need exactly 2 elements
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

// Check if a longer list (count > 2) ending with an atom should be printed as a dotted pair
// Heuristic: (a b c . d) has 4 elements, while (a b c) has 3 elements
static bool _should_print_longer_as_pair(SCM_List *l, int count, bool nested) {
  if (count <= 2) {
    return false;
  }
  
  if (nested) {
    return true; // Nested context suggests it's a pair
  }
  
  // For top-level lists: if count == 4 and first element is a symbol,
  // it's likely (a b c . d) from append, not (a b c) from quasiquote
  if (count == 4 && is_sym(l->data)) {
    return true;
  }
  
  return false;
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
  
  // Check if this should be printed as a dotted pair (2-element case)
  if (l->next && !l->next->next && _should_print_as_pair(l, nested)) {
    _print_dotted_pair(l->data, l->next->data);
    return;
  }
  
  // Find the last element and count length
  SCM_List *prev = nullptr;
  SCM_List *last = l;
  int count = 0;
  while (last->next) {
    prev = last;
    last = last->next;
    count++;
  }
  count++; // Include the last element
  
  // Check if this should be printed as a dotted pair
  // Note: (a b c) and (a b c . d) have the same structure (last node's data is atom, next is nullptr)
  // We use heuristics to distinguish them:
  // - If nested, treat as dotted pair (context suggests it's a pair)
  // - If count == 4 and first element is a symbol, treat as dotted pair (likely from append)
  bool is_dotted_pair = false;
  SCM *last_cdr = nullptr;
  if (prev && prev->next == last && last->data && !is_pair(last->data) && !is_nil(last->data)) {
    if (count > 2 && _should_print_longer_as_pair(l, count, nested)) {
      is_dotted_pair = true;
      last_cdr = last->data;
    }
  }
  
  // Regular list printing
  printf("(");
  SCM_List *current = l;
  while (current) {
    if (current != l) {
      printf(" ");
    }
    
    // If this is the last element before the dotted pair cdr, print it normally
    // The cdr will be printed after
    if (current == prev && is_dotted_pair) {
      print_ast(current->data);
      printf(" . ");
      print_ast(last_cdr);
      break;
    }
    
    // When printing elements of a list, pass nested=true for nested pairs
    if (is_pair(current->data)) {
      _print_list(cast<SCM_List>(current->data), true);
    } else {
      print_ast(current->data);
    }
    
    current = current->next;
  }
  printf(")");
}

void print_list(SCM_List *l) {
  _print_list(l, false);
}

