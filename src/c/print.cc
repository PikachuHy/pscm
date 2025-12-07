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
            _print_list(cast<SCM_List>(current->data), true);
          } else {
            print_ast(current->data);
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
            _print_list(cast<SCM_List>(cdr_current->data), true);
          } else {
            print_ast(cdr_current->data);
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
        print_ast(current->data);
        printf(" . ");
        print_ast(last->data);  // last->data is the cdr
        break;
      }
      
      // Print element
      if (is_pair(current->data)) {
        _print_list(cast<SCM_List>(current->data), true);
      } else {
        print_ast(current->data);
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
        _print_list(cast<SCM_List>(current->data), true);
      } else {
        print_ast(current->data);
      }
      
      current = current->next;
    }
    printf(")");
  }
}

void print_list(SCM_List *l) {
  _print_list(l, false);
}

