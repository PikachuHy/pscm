#include "pscm.h"

SCM_Environment::Entry *scm_env_search_entry(SCM_Environment *env, SCM_Symbol *sym, bool search_parent) {
  SCM_DEBUG_SYMTBL("search %p\n", env);
  auto l = env->dummy.next;
  while (l) {
    if (strcmp(l->data->key, sym->data) == 0) {
      SCM_DEBUG_SYMTBL("find %s\n", sym->data);
      if (debug_enabled) {
        printf("  ->  ");
        print_ast(l->data->value);
        printf("\n");
      }

      return l->data;
    }
    l = l->next;
  }
  if (search_parent && env->parent) {
    return scm_env_search_entry(env->parent, sym, search_parent);
  }
  return nullptr;
}

void scm_env_insert(SCM_Environment *env, SCM_Symbol *sym, SCM *value, bool search_parent) {
  if (value == nullptr) {
    printf("got it\n");
    exit(1);
  }
  auto entry = scm_env_search_entry(env, sym, search_parent);
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

SCM *scm_env_exist(SCM_Environment *env, SCM_Symbol *sym) {
  auto entry = scm_env_search_entry(env, sym);
  if (entry) {
    return entry->value;
  }
  return nullptr;
}

SCM *scm_env_search(SCM_Environment *env, SCM_Symbol *sym) {
  // 1. First search in lexical environment
  auto ret = scm_env_exist(env, sym);
  if (ret) {
    return ret;
  }
  
  // 2. If current module exists, search in module
  SCM *current_mod = scm_current_module();
  if (current_mod && is_module(current_mod)) {
    SCM_Module *module = cast<SCM_Module>(current_mod);
    SCM *var = module_search_variable(module, sym);
    if (var) {
      return var;  // Found (value can be #f)
    }
  }
  
  SCM_ERROR_SYMTBL("find %s, not found\n", sym->data);
  return nullptr;
}
