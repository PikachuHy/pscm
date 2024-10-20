#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdnoreturn.h>
#include <string.h>

#include "pscm/Parser.h"
#include "pscm/common_def.h"
import pscm;
using namespace pscm;
long *cont_base;

void print_basename(const char *path) {
  auto len = strlen(path);
  int i = len;
  while (i > 0) {
    if (path[i] == '/') {
      i++;
      break;
    }
    i--;
  }
  for (int idx = i; idx <= len; idx++) {
    printf("%c", path[idx]);
  }
}

#define SCM_FILENAME_LINENO()                                                                                          \
  print_basename(__BASE_FILE__);                                                                                       \
  printf(":%d ", __LINE__);

#define SCM_INIT_CONT(cont, base)                                                                                      \
  {                                                                                                                    \
    cont = new SCM_Continuation();                                                                                     \
    long __stack_top__;                                                                                                \
    cont->dst = &__stack_top__;                                                                                        \
    cont->stack_len = (long)base - (long)cont->dst;                                                                    \
    cont->stack_data = (long *)malloc(sizeof(long) * cont->stack_len);                                                 \
    memcpy((void *)cont->stack_data, (void *)cont->dst, sizeof(long) * cont->stack_len);                               \
  }                                                                                                                    \
  while (0)                                                                                                            \
    ;

#define SCM_APPLY_CONT(cont)                                                                                           \
  {                                                                                                                    \
    long __cur__;                                                                                                      \
    SCM_DEBUG_CONT("jump from %p to %p use %p\n", &__cur__, cont->dst, cont);                                          \
    memcpy(cont->dst, cont->stack_data, sizeof(long) * cont->stack_len);                                               \
    longjmp(cont->cont_jump_buffer, 1);                                                                                \
  }                                                                                                                    \
  while (0)                                                                                                            \
    ;

#define SCM_DEBUG(category, fmt, ...)                                                                                  \
  {                                                                                                                    \
    printf(category);                                                                                                  \
    printf(" ");                                                                                                       \
    SCM_FILENAME_LINENO();                                                                                             \
    printf(fmt, ##__VA_ARGS__);                                                                                        \
  }                                                                                                                    \
  while (0)                                                                                                            \
    ;

#define SCM_DEBUG_CONT(fmt, ...) SCM_DEBUG("[cont]", fmt, ##__VA_ARGS__)
#define SCM_DEBUG_SYMTBL(fmt, ...) SCM_DEBUG("[symtbl]", fmt, ##__VA_ARGS__)
#define SCM_DEBUG_TRANS(fmt, ...) SCM_DEBUG("[trans]", fmt, ##__VA_ARGS__)
#define SCM_DEBUG_EVAL(fmt, ...) SCM_DEBUG("[eval]", fmt, ##__VA_ARGS__)

#undef SCM_DEBUG_CONT
#define SCM_DEBUG_CONT(...)
#undef SCM_DEBUG_SYMTBL
#define SCM_DEBUG_SYMTBL(...)
#undef SCM_DEBUG_TRANS
#define SCM_DEBUG_TRANS(...)

struct SCM {
  enum Type { NONE, NIL, LIST, PROC, CONT, NUM, BOOL, SYM } type;

  void *value;
};

struct SCM_List {
  SCM *data;
  SCM_List *next;
};

struct SCM_Symbol {
  char *data;
  int len;
};

struct SCM_Procedure {
  SCM_Symbol *name;
  SCM_List *args;
  SCM_List *body;
};

struct SCM_Continuation {
  jmp_buf cont_jump_buffer;
  std::size_t stack_len;
  void *stack_data;
  void *dst;
  SCM *arg;
};

struct SCM_Environment {
  struct Entry {
    char *key;
    SCM *value;
  };

  struct List {
    Entry *data;
    List *next;
  };

  List dummy;
  SCM_Environment *parent;
};

SCM_Environment g_env;

SCM_Environment::Entry *scm_env_search_entry(SCM_Environment *env, SCM_Symbol *sym) {
  auto l = env->dummy.next;
  while (l) {
    if (strcmp(l->data->key, sym->data) == 0) {
      SCM_DEBUG_SYMTBL("find %s\n", sym->data);
      return l->data;
    }
    l = l->next;
  }
  if (env->parent) {
    return scm_env_search_entry(env->parent, sym);
  }
  return nullptr;
}

void scm_env_insert(SCM_Environment *env, SCM_Symbol *sym, SCM *value) {
  auto entry = scm_env_search_entry(env, sym);
  if (entry) {
    entry->value = value;
    return;
  }
  entry = new SCM_Environment::Entry();
  entry->key = new char[sym->len + 1];
  memcpy(entry->key, sym->data, sym->len);
  entry->value = value;
  auto node = new SCM_Environment::List();
  node->data = entry;
  node->next = env->dummy.next;
  env->dummy.next = node;
}

SCM *scm_env_search(SCM_Environment *env, SCM_Symbol *sym) {
  auto entry = scm_env_search_entry(env, sym);
  if (entry) {
    return entry->value;
  }
  printf("find %s, not found\n", sym->data);
  return nullptr;
}

SCM *scm_none_value = nullptr;

SCM *scm_nil_value = nullptr;

SCM *scm_none() {
  if (scm_none_value) {
    return scm_none_value;
  }
  scm_none_value = new SCM();
  scm_none_value->type = SCM::NONE;
  scm_none_value->value = nullptr;

  return scm_none_value;
}

bool is_none(SCM *scm) {
  return scm->type == SCM::NONE;
}

SCM *scm_nil() {
  if (scm_nil_value) {
    return scm_nil_value;
  }
  scm_nil_value = new SCM();
  scm_nil_value->type = SCM::NIL;
  scm_nil_value->value = nullptr;
  return scm_nil_value;
}

bool is_nil(SCM *scm) {
  return scm->type == SCM::NIL;
}

bool is_bool(SCM *scm) {
  return scm->type == SCM::BOOL;
}

SCM *create_sym(const char *data, int len) {
  SCM_Symbol *sym = new SCM_Symbol();
  sym->data = new char[len + 1];
  memcpy(sym->data, data, len);
  sym->len = len;
  SCM *scm = new SCM();
  scm->type = SCM::SYM;
  scm->value = sym;
  return scm;
}

bool is_sym(SCM *scm) {
  return scm->type == SCM::SYM;
}

bool is_pair(SCM *scm) {
  return scm->type == SCM::LIST;
}

bool is_num(SCM *scm) {
  return scm->type == SCM::NUM;
}

bool is_proc(SCM *scm) {
  return scm->type == SCM::PROC;
}

bool is_cont(SCM *scm) {
  return scm->type == SCM::CONT;
}

void print_ast(SCM *ast);
void print_list(SCM_List *l);

SCM *translate(Cell ret) {
  SCM_DEBUG_TRANS("translate %s\n", ret.to_std_string().c_str());
  if (ret.is_none()) {
    return scm_none();
  }
  if (!ret.is_pair()) {
    if (ret.is_sym()) {
      std::string sym_str;
      ret.to_sym()->to_string().toUTF8String(sym_str);
      auto sym = create_sym(sym_str.c_str(), sym_str.length());
      return sym;
    }

    printf("%s:%d [%s] not supported %s\n", __BASE_FILE__, __LINE__, __func__, car(ret).to_std_string().c_str());
    std::exit(1);
  }
  SCM_List dummy;
  dummy.data = nullptr;
  dummy.data = nullptr;
  SCM_List *it = &dummy;
  if (ret.is_pair()) {
    while (ret.is_pair()) {
      auto first = car(ret);
      if (first.is_pair()) {
        SCM_List *pair = new SCM_List();
        pair->data = translate(first);
        // print_ast(pair->data);
        // printf("\n");
        pair->next = nullptr;
        it->next = pair;
        it = pair;
        ret = cdr(ret);
      }
      else if (first.is_sym()) {
        std::string sym_str;
        car(ret).to_sym()->to_string().toUTF8String(sym_str);
        auto sym = create_sym(sym_str.c_str(), sym_str.length());
        SCM_List *pair = new SCM_List();
        pair->data = sym;
        pair->next = nullptr;
        it->next = pair;
        it = pair;
        ret = cdr(ret);
      }
      else if (first.is_nil()) {
        break;
      }
      else if (first.is_num()) {
        SCM_List *pair = new SCM_List();
        SCM *data = new SCM();
        data->type = SCM::NUM;
        auto val = first.to_num()->to_int();
        data->value = (void *)val;
        pair->data = data;
        it->next = pair;
        break;
      }
      else if (first.is_bool()) {
        SCM_List *pair = new SCM_List();
        SCM *data = new SCM();
        data->type = SCM::BOOL;
        auto val = first.to_bool();
        if (val) {
          data->value = (void *)1;
        }
        pair->data = data;
        it->next = pair;
        break;
      }
      else {
        printf("%s:%d [%s] not supported %s\n", __BASE_FILE__, __LINE__, __func__, car(ret).to_std_string().c_str());
        std::exit(1);
      }
    }
  }
  if (dummy.next) {
    SCM *l = new SCM();
    l->type = SCM::LIST;
    l->value = dummy.next;
    return l;
  }
  else {
    return scm_nil();
  }
}

SCM *parse(const std::string& code) {
  UString unicode(code.c_str());
  Parser parser(unicode);
  auto ret = parser.parse();
  return translate(ret);
}

void print_ast(SCM *ast) {
  if (is_proc(ast)) {
    auto proc = (SCM_Procedure *)ast->value;
    printf("<");
    printf("proc %s ", proc->name->data);
    print_list(proc->args);
    printf(">");
    return;
  }
  if (is_cont(ast)) {
    auto cont = (SCM_Continuation *)ast->value;
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
    SCM_Symbol *sym = (SCM_Symbol *)ast->value;
    printf("%s", sym->data);
    return;
  }
  if (is_pair(ast)) {
    SCM_List *l = (SCM_List *)ast->value;
    print_list(l);
    return;
  }
  if (is_nil(ast)) {
    printf("nil");
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
  printf("%s:%d not supported %d", __FILE__, __LINE__, ast->type);
  std::exit(1);
}

void print_list(SCM_List *l) {
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

SCM *eval_with_env(SCM_Environment *env, SCM *ast);

SCM *eval_with_list(SCM_Environment *env, SCM_List *l) {
  assert(l);
  SCM *ret = nullptr;
  while (l) {
    ret = eval_with_env(env, l->data);
    l = l->next;
  }
  return ret;
}

SCM *eval_with_env(SCM_Environment *env, SCM *ast) {
entry:
  if (!is_pair(ast)) {
    if (is_sym(ast)) {
      SCM_Symbol *sym = (SCM_Symbol *)ast->value;
      auto val = scm_env_search(env, sym);
      assert(val);
      return val;
    }
    return ast;
  }
  SCM_List *l = (SCM_List *)ast->value;
  assert(l->data);
  if (is_sym(l->data)) {
    SCM_Symbol *sym = (SCM_Symbol *)l->data->value;
    if (strcmp(sym->data, "define") == 0) {
      if (l->next && is_sym(l->next->data)) {
        SCM_Symbol *varname = (SCM_Symbol *)l->next->data->value;
        SCM_DEBUG_EVAL("define variable %s", varname->data);
        auto val = eval_with_env(env, l->next->next->data);
        if (is_proc(val)) {
          auto proc = (SCM_Procedure *)val->value;
          if (proc->name == nullptr) {
            proc->name = varname;
            SCM_DEBUG_EVAL("define proc from lambda\n");
          }
        }
        scm_env_insert(env, varname, val);
        // printf("NYI");
        // std::exit(0);
        return scm_nil();
      }
      else {
        SCM_List *proc_sig = (SCM_List *)l->next->data->value;
        SCM_DEBUG_EVAL("define a procedure");
        if (is_sym(proc_sig->data)) {
          SCM_Symbol *proc_name = (SCM_Symbol *)proc_sig->data->value;
          SCM_DEBUG_EVAL(" %s with params ", proc_name->data);
          printf("(");
          print_ast(proc_sig->next->data);
          printf(")\n");
          // handle procedure body
          SCM_Procedure *proc = new SCM_Procedure();
          proc->name = proc_name;
          proc->args = proc_sig->next;
          proc->body = l->next->next;

          SCM *ret = new SCM();
          ret->type = SCM::PROC;
          ret->value = proc;
          scm_env_insert(env, proc_name, ret);
          return ret;
        }
        else {
          printf("%s:%d not supported ", __FILE__, __LINE__);
          print_ast(proc_sig->data);
          printf("\n");
          std::exit(1);
        }
      }
    }
    else if (strcmp(sym->data, "call/cc") == 0) {
      SCM_Continuation *cont;
      SCM_INIT_CONT(cont, cont_base);
      int ret = setjmp(cont->cont_jump_buffer);
      if (ret) {
        SCM_DEBUG_CONT("jump back\n");
        // print_ast(cont->arg);
        // std::exit(0);
        return cont->arg;
      }
      else {
        SCM_DEBUG_CONT("after setjmp\n");
        auto arg = l->next->data;
        auto cont_arg = new SCM();
        cont_arg->type = SCM::CONT;
        cont_arg->value = cont;
        auto new_ast = new SCM_List();
        new_ast->data = eval_with_env(env, arg);
        auto new_ast_next = new SCM_List();
        new_ast_next->data = cont_arg;
        new_ast_next->next = nullptr;
        new_ast->next = new_ast_next;
        ast->type = SCM::LIST;
        ast->value = new_ast;
      }
      goto entry;
    }
    else if (strcmp(sym->data, "lambda") == 0) {
      SCM_List *proc_sig = (SCM_List *)l->next->data->value;
      SCM_Procedure *proc = new SCM_Procedure();
      proc->name = nullptr;
      proc->args = proc_sig;
      proc->body = l->next->next;

      SCM *ret = new SCM();
      ret->type = SCM::PROC;
      ret->value = proc;

      return ret;
    }
    else if (strcmp(sym->data, "set!") == 0) {
      assert(is_sym(l->next->data));
      auto sym = (SCM_Symbol *)l->next->data->value;
      auto val = eval_with_env(env, l->next->next->data);
      scm_env_insert(env, sym, val);
      return scm_nil();
    }
    else if (strcmp(sym->data, "quote") == 0) {
      assert(is_sym(l->next->data));
      return l->next->data;
    }
    else {
      auto val = scm_env_search(env, sym);
      if (!val) {
        printf("%s:%d Symbol not found '%s'\n", __FILE__, __LINE__, sym->data);
      }
      auto new_ast = new SCM();
      new_ast->type = SCM::LIST;
      auto new_list = new SCM_List();
      new_list->data = val;
      new_list->next = l->next;
      new_ast->value = new_list;
      ast = new_ast;
      return eval_with_env(env, ast);
    }
  }
  if (is_cont(l->data)) {
    auto cont = (SCM_Continuation *)l->data->value;
    cont->arg = eval_with_env(env, l->next->data);
    SCM_APPLY_CONT(cont);
  }
  else if (is_proc(l->data)) {
    auto proc = (SCM_Procedure *)l->data->value;
    auto proc_env = new SCM_Environment();
    proc_env->parent = env;
    proc_env->dummy.data = nullptr;
    proc_env->dummy.next = nullptr;
    auto args_l = proc->args;
    while (l->next && args_l) {
      assert(is_sym(args_l->data));
      auto arg_sym = (SCM_Symbol *)args_l->data->value;
      scm_env_insert(proc_env, arg_sym, l->next->data);
      l = l->next;
      args_l = args_l->next;
    }
    if (l && args_l) {
      printf("args not match\n");
      printf("expect ");
      print_list(proc->args);
      printf("\n");
      printf("but got ");
      print_list(l->next);
      printf("\n");
      std::exit(1);
    }
    auto val = eval_with_list(proc_env, proc->body);
    return val;
  }
  else {
    printf("%s:%d not supported ", __FILE__, __LINE__);
    print_list(l);
    printf("\n");
  }
  return scm_nil();
}

jmp_buf my_jump_buffer;
SCM *my_ret;
int my_count = 0;
bool debug = false;
long hook_base;
SCM_Continuation *my_cont;

SCM *eval(SCM *ast) {
  auto env = &g_env;
  long stack_base;
  cont_base = &stack_base;
  SCM_INIT_CONT(my_cont, hook_base);
  int ret = setjmp(my_cont->cont_jump_buffer);
  if (debug)
    for (int i = 0; i < 16; i++) {
      printf("%d %p\n", i, (void *)my_cont->cont_jump_buffer[i]);
    }
  if (ret != 0) {
    SCM_DEBUG_CONT("my jump\n");
    my_count++;
    // if (my_count == 5) {
    //   std::cout << "meet max count " << my_count << std::endl;
    //   std::exit(0);
    // }
    return my_ret;
  }
  else {
    auto ret = eval_with_env(env, ast);
    my_ret = ret;
    SCM_APPLY_CONT(my_cont);
  }
}

void init_scm() {
  g_env.parent = nullptr;
  g_env.dummy.data = nullptr;
  g_env.dummy.next = nullptr;
}

#define SCM_PRINT_AST(ast)                                                                                             \
  printf("[ast] ");                                                                                                    \
  print_ast(ast);                                                                                                      \
  printf("\n");

void my_eval(SCM *ast) {
  SCM_PRINT_AST(ast);
  auto val = eval(ast);
  printf(" --> ");
  SCM_PRINT_AST(val);
}

void run1() {
  SCM *ast = parse(R"(
  (define (f return)
    (return 2)
    3)
  )");
  my_eval(ast);

  ast = parse(R"(
(call/cc f)
  )");
  my_eval(ast);
}

void run2() {
  SCM *ast = parse(R"(
(define c #f)
  )");
  my_eval(ast);

  ast = parse(R"(
(call/cc
  (lambda (c0)
          (set! c c0)
          'talk1))
  )");
  my_eval(ast);

  ast = parse(R"(
(c 'talk2)
  )");
  my_eval(ast);

  ast = parse(R"(
(c 'talk2)
  )");
  my_eval(ast);
}

void repl() {
  char buffer[100];
  while (true) {
    printf("pscm> ");
    if (fgets(buffer, sizeof(buffer), stdin) != NULL) {
      auto ast = parse(buffer);
      print_ast(ast);
      printf("\n");
      auto val = eval(ast);
      printf(" --> ");
      print_ast(val);
      printf("\n");
    }
    else {
      printf("read failed\n");
      std::exit(1);
    }
  }
}

int main() {
  long t;
  hook_base = (long)&t;
  init_scm();
  run1();

  run2();
  // repl();
  return 0;
}

/*
(define c #f)
(call/cc (lambda (c0) (set! c c0) 'talk1))
(c 'talk2)
*/