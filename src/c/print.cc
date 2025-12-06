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
  printf("%s:%d not supported %d\n", __FILE__, __LINE__, ast->type);
  exit(1);
}

void print_list(SCM_List *l) {
  // handle quote
  if (l && is_sym(l->data)) {
    auto sym = cast<SCM_Symbol>(l->data);
    if (strcmp(sym->data, "quote") == 0) {
      printf("'");
      if (l->next) {
        print_ast(l->next->data);
        assert(!l->next->next);
      }
      else {
        printf("()");
      }
      return;
    }
  }
  printf("(");
  while (l) {
    print_ast(l->data);
    l = l->next;
    if (l) {
      printf(" ");
    }
  }
  printf(")");
}
