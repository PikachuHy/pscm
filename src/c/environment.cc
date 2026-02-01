#include "pscm.h"
#include "eval.h"  // For g_current_eval_context and get_source_location_str

SCM_Environment::Entry *scm_env_search_entry(SCM_Environment *env, SCM_Symbol *sym, bool search_parent) {
  SCM_DEBUG_SYMTBL("search %p\n", env);
  auto l = env->dummy.next;
  while (l) {
    if (l->data && l->data->key && sym && sym->data && strcmp(l->data->key, sym->data) == 0) {
      SCM_DEBUG_SYMTBL("find %s\n", sym->data);
      if (debug_enabled) {
        printf("  ->  ");
        if (l->data->value) {
          print_ast(l->data->value);
        } else {
          printf("<null>");
        }
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
  auto entry = scm_env_search_entry(env, sym, search_parent);
  if (entry) {
    entry->value = value;
    return;
  }
  entry = new SCM_Environment::Entry();
  entry->key = new char[sym->len + 1];
  memcpy(entry->key, sym->data, sym->len);
  entry->key[sym->len] = '\0';  // Ensure null termination
  entry->value = value;
  auto node = new SCM_Environment::List();
  node->data = entry;
  node->next = env->dummy.next;
  env->dummy.next = node;
}

SCM *scm_env_exist(SCM_Environment *env, SCM_Symbol *sym) {
  // Search only in current environment (no parent search)
  // This ensures local bindings take precedence over parent bindings
  auto entry = scm_env_search_entry(env, sym, /*search_parent=*/false);
  if (entry) {
    return entry->value;  // Value can be null (e.g., for #f)
  }
  return nullptr;
}

SCM *scm_env_search(SCM_Environment *env, SCM_Symbol *sym) {
  // 1. First search in lexical environment (local bindings, no parent search)
  // This ensures local bindings in let/lambda take precedence
  auto ret = scm_env_exist(env, sym);
  if (ret) {
    return ret;
  }
  
  // 2. Search in parent environments (let/lambda bindings)
  // This must come before module search to ensure lexical scoping works correctly
  // Parent environments (from let/lambda) should take precedence over module bindings
  auto entry = scm_env_search_entry(env, sym, /*search_parent=*/true);
  if (entry) {
    return entry->value;  // Value can be null (e.g., for #f)
  }
  
  // 3. If current module exists, search in module (module bindings override global)
  // This is important: when define is used in a module, it should shadow global bindings
  // But module bindings come after lexical bindings (let/lambda)
  SCM *current_mod = scm_current_module();
  if (current_mod && is_module(current_mod)) {
    SCM_Module *module = cast<SCM_Module>(current_mod);
    SCM *var = module_search_variable(module, sym);
    if (var) {
      return var;  // Found (value can be #f)
    }
  }
  
  // 4. Finally, search in global environment (g_env) if we haven't found it yet
  // This ensures built-in functions registered via scm_define_function are accessible
  // This is needed because when env != &g_env, we need to check g_env for built-ins
  // When env == &g_env, we've already searched it in step 1 and 2, so this is a fallback
  // for cases where the symbol might be in g_env but not found in the current search path
  if (env != &g_env) {
    auto global_entry = scm_env_search_entry(&g_env, sym, /*search_parent=*/false);
    if (global_entry) {
      return global_entry->value;  // Value can be null (e.g., for #f)
    }
  }
  
  // Try to get source location from current eval context for debugging
  const char *loc_str = nullptr;
  if (g_current_eval_context) {
    loc_str = get_source_location_str(g_current_eval_context);
  }
  
  if (loc_str) {
    SCM_ERROR_SYMTBL("find %s, not found (at %s)\n", sym->data, loc_str);
  } else {
    SCM_ERROR_SYMTBL("find %s, not found\n", sym->data);
  }
  return nullptr;
}
