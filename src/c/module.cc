#include "pscm.h"
#include "eval.h"

// Current module (using simple global variable, can be changed to fluid in the future)
static SCM *g_current_module = nullptr;
SCM *g_root_module = nullptr;  // Root module (pscm-user), exported for use in other files

// Module registry: module name -> module object
static SCM_HashTable *g_module_registry = nullptr;

// Create a module
SCM *scm_make_module(SCM_List *name, int obarray_size) {
  auto module = new SCM_Module();
  
  // Create obarray (hash table)
  SCM *size_scm = new SCM();
  size_scm->type = SCM::NUM;
  size_scm->value = (void *)(int64_t)obarray_size;
  size_scm->source_loc = nullptr;
  SCM *obarray_scm = scm_c_make_hash_table(size_scm);
  module->obarray = cast<SCM_HashTable>(obarray_scm);
  
  // Initialize other fields
  module->uses = nullptr;
  module->binder = nullptr;
  module->eval_closure = nullptr;
  module->transformer = nullptr;
  module->name = name;
  module->kind = nullptr;  // Default to 'module, can be set later
  module->public_interface = nullptr;
  module->exports = nullptr;
  
  return wrap(module);
}

// Get current module
SCM *scm_current_module() {
  if (g_current_module) {
    return g_current_module;
  }
  // If no current module, return root module
  if (!g_root_module) {
    // Initialize root module
    SCM_List *name = make_list(wrap(make_sym("pscm-user")));
    g_root_module = scm_make_module(name, 31);
    g_current_module = g_root_module;
  }
  return g_root_module;
}

// Set current module
SCM *scm_set_current_module(SCM *module) {
  if (!is_module(module)) {
    eval_error("set-current-module: expected module");
  }
  SCM *old = scm_current_module();
  g_current_module = module;
  return old;
}

// Helper function to lookup variable in module's obarray
// Returns the value if found (can be #f), or nullptr if not found
// This allows us to distinguish between "not found" and "value is #f"
SCM *module_obarray_lookup(SCM_Module *module, SCM_Symbol *sym) {
  
  SCM *handle = scm_c_hash_get_handle_eq(wrap(module->obarray), wrap(sym));
  // hash-get-handle returns #f if key not found, or (key . value) if found
  // We check if handle is #f (not found) vs. a pair (found, value could be #f)
  if (handle && handle != scm_bool_false() && is_pair(handle)) {
    // Handle exists (it's a pair, not #f)
    // handle is the entry pair (key . value) wrapped
    auto pair = cast<SCM_List>(handle);
    // pair->data is the key, pair->next->data is the value
    if (pair->next) {
      return pair->next->data;  // Return value (can be #f)
    } else {
      // value is nil (entry->next is nullptr)
      return scm_nil();
    }
  }
  
  return nullptr;  // Not found
}

// Helper function to search variable in module (obarray, uses list, and root module)
// Returns the value if found (can be #f), or nullptr if not found
// This allows us to distinguish between "not found" and "value is #f"
// Does not call binder (unlike scm_module_variable)
SCM *module_search_variable(SCM_Module *module, SCM_Symbol *sym) {
  // 1. Check module obarray
  SCM *var = module_obarray_lookup(module, sym);
  if (var) {
    return var;  // Found (value can be #f)
  }
  
  // 2. Search uses list
  SCM_List *uses = module->uses;
  while (uses) {
    SCM *use_module_scm = uses->data;
    if (is_module(use_module_scm)) {
      SCM_Module *use_module = cast<SCM_Module>(use_module_scm);
      var = module_obarray_lookup(use_module, sym);
      if (var) {
        return var;  // Found (value can be #f)
      }
    }
    uses = uses->next;
  }
  
  // 3. If not found, also search root module (pscm-user) for global variables like %load-path
  if (g_root_module && module != cast<SCM_Module>(g_root_module)) {
    SCM_Module *root_module = cast<SCM_Module>(g_root_module);
    var = module_obarray_lookup(root_module, sym);
    if (var) {
      return var;  // Found (value can be #f)
    }
  }
  
  return nullptr;  // Not found
}

// Helper function to find which module contains a variable (for updating)
// Returns the module that contains the variable, or nullptr if not found
// Searches: current module obarray, uses list, and root module
SCM_Module *module_find_variable_module(SCM_Module *module, SCM_Symbol *sym) {
  // 1. Check module obarray
  if (module_obarray_lookup(module, sym)) {
    return module;  // Found in current module
  }
  
  // 2. Search uses list
  SCM_List *uses = module->uses;
  while (uses) {
    SCM *use_module_scm = uses->data;
    if (is_module(use_module_scm)) {
      SCM_Module *use_module = cast<SCM_Module>(use_module_scm);
      if (module_obarray_lookup(use_module, sym)) {
        return use_module;  // Found in use module
      }
    }
    uses = uses->next;
  }
  
  // 3. Check root module
  if (g_root_module && module != cast<SCM_Module>(g_root_module)) {
    SCM_Module *root_module = cast<SCM_Module>(g_root_module);
    if (module_obarray_lookup(root_module, sym)) {
      return root_module;  // Found in root module
    }
  }
  
  return nullptr;  // Not found
}

// Look up variable in module
SCM *scm_module_variable(SCM_Module *module, SCM_Symbol *sym, bool definep) {
  // 1. Check module obarray, uses list, and root module
  SCM *var = module_search_variable(module, sym);
  if (var) {
    return var;  // Found (value can be #f)
  }
  
  // 2. Call binder (if exists and not a define operation)
  if (!definep && module->binder) {
    // TODO: Implement binder call
    // Skip for now
  }
  
  return scm_bool_false();  // #f (not found)
}

// Helper function to convert module name to file path
// e.g., (test) -> "test.scm"
// e.g., (ice-9 common-list) -> "ice-9/common-list.scm"
// Returns allocated string, caller must free it
static char *module_name_to_path(SCM_List *name) {
  // First pass: calculate total length needed
  size_t total_len = 0;
  SCM_List *current = name;
  bool first = true;
  while (current && current->data) {
    if (!first) {
      total_len += 1; // "/"
    }
    first = false;
    if (is_sym(current->data)) {
      SCM_Symbol *sym = cast<SCM_Symbol>(current->data);
      total_len += sym->len;
    }
    current = current->next;
  }
  total_len += 5; // ".scm" + null terminator
  
  // Allocate buffer
  char *path = (char *)malloc(total_len);
  if (!path) {
    return nullptr;
  }
  
  // Second pass: build the path
  char *p = path;
  current = name;
  first = true;
  while (current && current->data) {
    if (!first) {
      *p++ = '/';
    }
    first = false;
    if (is_sym(current->data)) {
      SCM_Symbol *sym = cast<SCM_Symbol>(current->data);
      memcpy(p, sym->data, sym->len);
      p += sym->len;
    }
    current = current->next;
  }
  memcpy(p, ".scm", 4);
  p += 4;
  *p = '\0';
  
  return path;
}

// Helper function to check if file exists
static bool file_exists(const char *path) {
  FILE *f = fopen(path, "r");
  if (f) {
    fclose(f);
    return true;
  }
  return false;
}

// Resolve module name, return module object
SCM *scm_resolve_module(SCM_List *name) {
  // Initialize registry if needed
  if (!g_module_registry) {
    g_module_registry = new SCM_HashTable();
    g_module_registry->capacity = 61;
    g_module_registry->size = 0;
    g_module_registry->buckets = (SCM **)calloc(61, sizeof(SCM *));
    for (size_t i = 0; i < 61; i++) {
      g_module_registry->buckets[i] = scm_nil();
    }
  }
  
  // 1. Check registry
  SCM *name_key = wrap(name);
  // Use equal? comparison for module names (lists), not eq?
  SCM *module = scm_c_hash_ref_equal(wrap(g_module_registry), name_key);
  // scm_c_hash_ref_equal returns #f if not found, so check for #f explicitly
  if (module && !is_falsy(module) && is_module(module)) {
    return module;
  }
  
  // 2. Try to load from file system
  // Convert module name to file path
  char *module_path = module_name_to_path(name);
  if (!module_path) {
    // If path conversion failed, create empty module
    module = scm_make_module(name, 31);
    scm_c_hash_set_equal(wrap(g_module_registry), wrap(name), module);
    return module;
  }
  
  // Get %load-path from root module
  SCM *load_path = nullptr;
  if (g_root_module) {
    SCM_Module *root_module = cast<SCM_Module>(g_root_module);
    SCM_Symbol *load_path_sym = make_sym("%load-path");
    load_path = module_obarray_lookup(root_module, load_path_sym);
  }
  
  // Search in %load-path
  const char *found_path = nullptr;
  char *full_path = nullptr;
  if (load_path && is_pair(load_path)) {
    SCM_List *path_list = cast<SCM_List>(load_path);
    while (path_list && path_list->data) {
      if (is_str(path_list->data)) {
        SCM_String *path_str = cast<SCM_String>(path_list->data);
        size_t path_str_len = path_str->len;
        size_t module_path_len = strlen(module_path);
        
        // Calculate total length: path_str + "/" (if needed) + module_path + null
        size_t total_len = path_str_len + module_path_len + 2;
        if (path_str_len > 0 && path_str->data[path_str_len - 1] != '/') {
          total_len += 1; // Need to add "/"
        }
        
        char *full = (char *)malloc(total_len);
        if (full) {
          memcpy(full, path_str->data, path_str_len);
          char *p = full + path_str_len;
          if (path_str_len > 0 && path_str->data[path_str_len - 1] != '/') {
            *p++ = '/';
          }
          memcpy(p, module_path, module_path_len);
          p += module_path_len;
          *p = '\0';
          
          if (file_exists(full)) {
            found_path = full;
            full_path = full;
            break;
          } else {
            free(full);
          }
        }
      }
      path_list = path_list->next;
    }
  }
  
  // If not found in %load-path, try current directory
  if (!found_path) {
    if (file_exists(module_path)) {
      size_t len = strlen(module_path);
      full_path = (char *)malloc(len + 1);
      if (full_path) {
        strcpy(full_path, module_path);
        found_path = full_path;
      }
    }
  }
  
  // Free module_path (no longer needed)
  free(module_path);
  
  // Create module (before loading, so define-module can find it)
  module = scm_make_module(name, 31);
  
  // Register module (before loading, so define-module can register it)
  scm_c_hash_set_equal(wrap(g_module_registry), wrap(name), module);
  
  // If file found, load it
  if (found_path) {
    // Save current module
    SCM *old_module = scm_current_module();
    
    // Set current module to the new module
    scm_set_current_module(module);
    
    // Load and evaluate file
    SCM_List *expr_list = parse_file(found_path);
    if (expr_list) {
      SCM_Environment *top_env = g_env.parent ? g_env.parent : &g_env;
      SCM_List *it = expr_list;
      while (it) {
        eval_with_env(top_env, it->data);
        it = it->next;
      }
    }
    
    // Restore current module
    scm_set_current_module(old_module);
    
    if (full_path) {
      free(full_path);
    }
  }
  
  return module;
}

// Helper function to export symbol
void scm_module_export(SCM_Module *module, SCM_Symbol *sym) {
  // Add to export list
  SCM_List *new_export = make_list(wrap(sym));
  if (module->exports) {
    new_export->next = module->exports;
  }
  module->exports = new_export;
  
  // Update public interface (simplified implementation, don't create separate public interface module for now)
  // TODO: Implement complete public interface mechanism
}

// Update module's public interface
void scm_update_module_public_interface(SCM_Module *module) {
  // Simplified implementation: don't create separate public interface module for now
  // Future: can create a new module containing only exported bindings
  // TODO: Implement complete public interface mechanism
}

// C API 函数
SCM *scm_c_current_module(SCM *arg) {
  (void)arg;  // Unused
  return scm_current_module();
}

SCM *scm_c_set_current_module(SCM *module) {
  return scm_set_current_module(module);
}

SCM *scm_c_resolve_module(SCM *name_arg) {
  if (!name_arg) {
    eval_error("resolve-module: expected module name");
  }
  if (!is_pair(name_arg)) {
    eval_error("resolve-module: module name must be a list");
  }
  return scm_resolve_module(cast<SCM_List>(name_arg));
}

SCM *scm_c_module_ref(SCM_List *args) {
  if (!args || !args->data) {
    eval_error("module-ref: requires at least 2 arguments");
    return nullptr;
  }
  
  SCM *module = args->data;
  if (!is_module(module)) {
    eval_error("module-ref: expected module");
    return nullptr;
  }
  
  if (!args->next || !args->next->data) {
    eval_error("module-ref: expected symbol");
    return nullptr;
  }
  
  SCM *symbol = args->next->data;
  if (!is_sym(symbol)) {
    eval_error("module-ref: expected symbol");
    return nullptr;
  }
  
  // Get optional default value
  SCM *default_val = nullptr;
  if (args->next->next && args->next->next->data) {
    default_val = args->next->next->data;
  }
  
  SCM_Module *mod = cast<SCM_Module>(module);
  SCM_Symbol *sym = cast<SCM_Symbol>(symbol);
  
  // Special handling for %module-public-interface
  if (strcmp(sym->data, "%module-public-interface") == 0) {
    // Return the public interface module, or the module itself if no public interface exists
    if (mod->public_interface) {
      return wrap(mod->public_interface);
    } else {
      // If no public interface, return the module itself (Guile behavior)
      return module;
    }
  }
  
  // Search module (obarray, uses list, and root module)
  SCM *var = module_search_variable(mod, sym);
  if (var) {
    return var;  // Found (value can be #f)
  }
  
  // Variable not found, return default value if provided
  if (default_val) {
    return default_val;
  }
  
  eval_error("module-ref: symbol '%s' not found in module", sym->data);
  return nullptr;
}

SCM *scm_c_module_bound_p(SCM *module, SCM *symbol) {
  if (!is_module(module)) {
    eval_error("module-bound?: expected module");
  }
  if (!is_sym(symbol)) {
    eval_error("module-bound?: expected symbol");
  }
  
  SCM_Module *mod = cast<SCM_Module>(module);
  SCM_Symbol *sym = cast<SCM_Symbol>(symbol);
  SCM *var = scm_module_variable(mod, sym, false);
  
  return var && !is_falsy(var) ? scm_bool_true() : scm_bool_false();
}

SCM *scm_c_module_p(SCM *obj) {
  return is_module(obj) ? scm_bool_true() : scm_bool_false();
}

// define-module special form
SCM *eval_define_module(SCM_Environment *env, SCM_List *l) {
  // Syntax: (define-module name [options ...])
  if (!l || !l->next || !l->next->data) {
    eval_error("define-module: missing module name");
  }
  
  SCM *name_scm = l->next->data;
  if (!is_pair(name_scm)) {
    eval_error("define-module: module name must be a list");
  }
  
  SCM_List *name = cast<SCM_List>(name_scm);
  
  // Parse options (simplified version, only handle basic options)
  SCM_List *options = l->next->next;
  
  // Check if module already exists in registry (e.g., created by scm_resolve_module)
  if (!g_module_registry) {
    g_module_registry = new SCM_HashTable();
    g_module_registry->capacity = 61;
    g_module_registry->size = 0;
    g_module_registry->buckets = (SCM **)calloc(61, sizeof(SCM *));
    for (size_t i = 0; i < 61; i++) {
      g_module_registry->buckets[i] = scm_nil();
    }
  }
  
  // Use equal? comparison for module names (lists), not eq?
  SCM *existing = scm_c_hash_ref_equal(wrap(g_module_registry), wrap(name));
  if (existing && !is_falsy(existing) && is_module(existing)) {
    // Module already exists (e.g., created by scm_resolve_module), use it
    scm_set_current_module(existing);
    return scm_none();
  }
  
  // Create new module
  SCM *module_scm = scm_make_module(name, 31);
  
  // Process options (simplified handling, need to parse #:use-module, #:export, etc.)
  // TODO: Implement complete option parsing
  
  // Register module
  // Use equal? comparison for module names (lists), not eq?
  scm_c_hash_set_equal(wrap(g_module_registry), wrap(name), module_scm);
  
  // Set current module
  scm_set_current_module(module_scm);
  
  return scm_none();
}

// use-modules special form
SCM *eval_use_modules(SCM_Environment *env, SCM_List *l) {
  // Syntax: (use-modules spec ...)
  SCM_Module *current = cast<SCM_Module>(scm_current_module());
  
  // l is (use-modules spec ...), so specs start from l->next
  SCM_List *specs = l->next;
  while (specs) {
    SCM *spec_scm = specs->data;
    // spec can be a module name list, e.g. (ice-9 common-list)
    if (is_pair(spec_scm)) {
      SCM_List *name = cast<SCM_List>(spec_scm);
      SCM *module_scm = scm_resolve_module(name);
      SCM_Module *module = cast<SCM_Module>(module_scm);
      
      // Get module's public interface
      SCM_Module *interface = module->public_interface;
      if (!interface) {
        // If no public interface, use module itself
        interface = module;
      }
      
      // Add to uses list
      SCM_List *new_use = make_list(wrap(interface));
      if (current->uses) {
        new_use->next = current->uses;
      }
      current->uses = new_use;
    }
    specs = specs->next;
  }
  
  return scm_none();
}

// export special form
SCM *eval_export(SCM_Environment *env, SCM_List *l) {
  // Syntax: (export symbol ...)
  SCM_Module *module = cast<SCM_Module>(scm_current_module());
  
  // l is (export symbol ...), so symbols start from l->next
  SCM_List *symbols = l->next;
  while (symbols) {
    SCM *sym_scm = symbols->data;
    if (is_sym(sym_scm)) {
      scm_module_export(module, cast<SCM_Symbol>(sym_scm));
    }
    symbols = symbols->next;
  }
  
  return scm_none();
}

// re-export special form
// Re-export symbols from other modules (typically imported via use-modules)
SCM *eval_re_export(SCM_Environment *env, SCM_List *l) {
  // Syntax: (re-export symbol ...)
  // Similar to export, but used to re-export symbols from other modules
  SCM_Module *module = cast<SCM_Module>(scm_current_module());
  
  // l is (re-export symbol ...), so symbols start from l->next
  SCM_List *symbols = l->next;
  while (symbols) {
    SCM *sym_scm = symbols->data;
    if (is_sym(sym_scm)) {
      // Check if symbol is available in current module (from uses or obarray)
      SCM_Symbol *sym = cast<SCM_Symbol>(sym_scm);
      SCM *var = module_search_variable(module, sym);
      if (!var) {
        // Symbol not found, but we still add it to exports
        // This allows re-exporting symbols that will be available later
        // (e.g., from modules that will be used)
        // In a strict implementation, we might want to error here
        // For now, we allow it for flexibility
      }
      // Add to export list regardless
      scm_module_export(module, sym);
    }
    symbols = symbols->next;
  }
  
  return scm_none();
}

// define-public special form
SCM *eval_define_public(SCM_Environment *env, SCM_List *l) {
  // Syntax: (define-public name value) or (define-public (name args ...) body ...)
  // First call regular define
  SCM *result = eval_define(env, l);
  
  // Then export symbol
  if (l->next) {
    SCM *name_scm = l->next->data;
    if (is_pair(name_scm)) {
      // (name args ...) form
      SCM_List *name_list = cast<SCM_List>(name_scm);
      if (name_list && name_list->data && is_sym(name_list->data)) {
        name_scm = name_list->data;
      }
    }
    
    if (is_sym(name_scm)) {
      SCM_Symbol *sym = cast<SCM_Symbol>(name_scm);
      scm_module_export(cast<SCM_Module>(scm_current_module()), sym);
    }
  }
  
  return result;
}

// module-map: Apply procedure to each binding in module
// (module-map proc module) -> list of results
SCM *scm_c_module_map(SCM_List *args) {
  if (!args || !args->data) {
    eval_error("module-map: requires at least 2 arguments");
    return nullptr;
  }
  
  SCM *proc = args->data;
  if (!is_proc(proc) && !is_func(proc)) {
    eval_error("module-map: first argument must be a procedure");
    return nullptr;
  }
  
  if (!args->next || !args->next->data) {
    eval_error("module-map: requires module argument");
    return nullptr;
  }
  
  SCM *module = args->next->data;
  if (!is_module(module)) {
    eval_error("module-map: second argument must be a module");
    return nullptr;
  }
  
  SCM_Module *mod = cast<SCM_Module>(module);
  
  // Collect results by iterating over module's obarray
  SCM_List *result_head = nullptr;
  SCM_List *result_tail = nullptr;
  
  // Iterate over all buckets in the hash table
  for (size_t i = 0; i < mod->obarray->capacity; i++) {
    SCM *bucket = mod->obarray->buckets[i];
    if (bucket && !is_nil(bucket)) {
      auto l = cast<SCM_List>(bucket);
      while (l) {
        if (is_pair(l->data)) {
          auto entry = cast<SCM_List>(l->data);
          if (entry->data && entry->next) {
            SCM *key = entry->data;  // Symbol
            SCM *value = entry->next->data;  // Variable value
            
            // Call proc with (key value)
            // Build call expression: (proc key value)
            SCM_List args_dummy = make_list_dummy();
            args_dummy.data = proc;
            auto args_tail = &args_dummy;
            
            // Wrap key in quote to prevent re-evaluation
            SCM *quoted_key = scm_list2(scm_sym_quote(), key);
            args_tail->next = make_list(quoted_key);
            args_tail = args_tail->next;
            
            // Wrap value in quote
            SCM *quoted_value = scm_list2(scm_sym_quote(), value);
            args_tail->next = make_list(quoted_value);
            
            // Build and evaluate call expression
            SCM call_expr;
            call_expr.type = SCM::LIST;
            call_expr.value = &args_dummy;
            call_expr.source_loc = nullptr;  // Mark as temporary to skip call stack tracking
            
            SCM *proc_result = eval_with_env(&g_env, &call_expr);
            
            // Append result to result list
            SCM_List *new_node = make_list(proc_result);
            if (!result_head) {
              result_head = new_node;
              result_tail = new_node;
            } else {
              result_tail->next = new_node;
              result_tail = new_node;
            }
          }
        }
        l = l->next;
      }
    }
  }
  
  return result_head ? wrap(result_head) : scm_nil();
}

// module-use!: Add module to uses list
// (module-use! module spec) -> unspecified
SCM *scm_c_module_use(SCM_List *args) {
  if (!args || !args->data) {
    eval_error("module-use!: requires at least 2 arguments");
    return nullptr;
  }
  
  SCM *module = args->data;
  if (!is_module(module)) {
    eval_error("module-use!: first argument must be a module");
    return nullptr;
  }
  
  if (!args->next || !args->next->data) {
    eval_error("module-use!: requires module spec argument");
    return nullptr;
  }
  
  SCM *spec = args->next->data;
  SCM_Module *mod = cast<SCM_Module>(module);
  
  // spec can be a module name list or a module object
  SCM *use_module = nullptr;
  if (is_pair(spec)) {
    // spec is a module name list, resolve it
    SCM_List *name = cast<SCM_List>(spec);
    use_module = scm_resolve_module(name);
  } else if (is_module(spec)) {
    // spec is already a module object
    use_module = spec;
  } else {
    eval_error("module-use!: second argument must be a module name list or module object");
    return nullptr;
  }
  
  if (!use_module || !is_module(use_module)) {
    eval_error("module-use!: failed to resolve module");
    return nullptr;
  }
  
  SCM_Module *use_mod = cast<SCM_Module>(use_module);
  
  // Get module's public interface
  SCM_Module *interface = use_mod->public_interface;
  if (!interface) {
    // If no public interface, use module itself
    interface = use_mod;
  }
  
  // Add to uses list
  SCM_List *new_use = make_list(wrap(interface));
  if (mod->uses) {
    new_use->next = mod->uses;
  }
  mod->uses = new_use;
  
  return scm_none();
}

void init_modules() {
  // Create root module
  SCM_List *name = make_list(wrap(make_sym("pscm-user")));
  g_root_module = scm_make_module(name, 31);
  g_current_module = g_root_module;
  
  // Initialize module registry
  g_module_registry = new SCM_HashTable();
  g_module_registry->capacity = 61;
  g_module_registry->size = 0;
  g_module_registry->buckets = (SCM **)calloc(61, sizeof(SCM *));
  for (size_t i = 0; i < 61; i++) {
    g_module_registry->buckets[i] = scm_nil();
  }
  
  // Register root module
  // Use equal? comparison for module names (lists), not eq?
  scm_c_hash_set_equal(wrap(g_module_registry), wrap(name), g_root_module);
  
  // Register built-in functions
  scm_define_function("current-module", 0, 0, 0, scm_c_current_module);
  scm_define_function("set-current-module", 1, 0, 0, scm_c_set_current_module);
  scm_define_function("resolve-module", 1, 0, 0, scm_c_resolve_module);
  scm_define_vararg_function("module-ref", scm_c_module_ref);
  scm_define_function("module-bound?", 2, 0, 0, scm_c_module_bound_p);
  scm_define_function("module?", 1, 0, 0, scm_c_module_p);
  scm_define_vararg_function("module-map", scm_c_module_map);
  scm_define_vararg_function("module-use!", scm_c_module_use);
}

