#include "pscm.h"
#include "eval.h"

// Forward declarations for hash table functions
extern SCM *scm_c_make_hash_table(SCM *size_arg);
extern SCM *scm_c_hash_ref_eq(SCM *table, SCM *key);
extern SCM *scm_c_hash_set_eq(SCM *table, SCM *key, SCM *value);

// Current module (using simple global variable, can be changed to fluid in the future)
static SCM *g_current_module = nullptr;
static SCM *g_root_module = nullptr;  // Root module (pscm-user)

// Module registry: module name -> module object
static SCM_HashTable *g_module_registry = nullptr;

// Forward declarations
SCM *scm_make_module(SCM_List *name, int obarray_size);
SCM *scm_current_module();
SCM *scm_set_current_module(SCM *module);
SCM *scm_module_variable(SCM_Module *module, SCM_Symbol *sym, bool definep);
SCM *scm_resolve_module(SCM_List *name);
void scm_module_export(SCM_Module *module, SCM_Symbol *sym);

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

// Look up variable in module
SCM *scm_module_variable(SCM_Module *module, SCM_Symbol *sym, bool definep) {
  // 1. Check module obarray
  SCM *var = scm_c_hash_ref_eq(wrap(module->obarray), wrap(sym));
  if (var && !is_falsy(var)) {
    return var;
  }
  
  // 2. Call binder (if exists and not a define operation)
  if (!definep && module->binder) {
    // TODO: Implement binder call
    // Skip for now
  }
  
  // 3. Search uses list
  SCM_List *uses = module->uses;
  while (uses) {
    SCM *use_module_scm = uses->data;
    if (is_module(use_module_scm)) {
      SCM_Module *use_module = cast<SCM_Module>(use_module_scm);
      SCM *var = scm_module_variable(use_module, sym, false);
      if (var && !is_falsy(var)) {
        return var;
      }
    }
    uses = uses->next;
  }
  
  return scm_bool_false();  // #f
}

// Resolve module name, return module object
SCM *scm_resolve_module(SCM_List *name) {
  // 1. Check registry
  if (g_module_registry) {
    SCM *name_key = wrap(name);
    SCM *module = scm_c_hash_ref_eq(wrap(g_module_registry), name_key);
    if (module && is_module(module)) {
      return module;
    }
  }
  
  // 2. Try to load from file system (future implementation)
  // For now, create a new module
  
  // Create a new module temporarily (simplified implementation)
  SCM *module = scm_make_module(name, 31);
  
  // Register module
  if (!g_module_registry) {
    g_module_registry = new SCM_HashTable();
    g_module_registry->capacity = 61;
    g_module_registry->size = 0;
    g_module_registry->buckets = (SCM **)calloc(61, sizeof(SCM *));
    for (size_t i = 0; i < 61; i++) {
      g_module_registry->buckets[i] = scm_nil();
    }
  }
  scm_c_hash_set_eq(wrap(g_module_registry), wrap(name), module);
  
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
SCM *scm_c_current_module(SCM_List *args) {
  return scm_current_module();
}

SCM *scm_c_set_current_module(SCM *module) {
  return scm_set_current_module(module);
}

SCM *scm_c_resolve_module(SCM_List *name) {
  if (!name || !name->data) {
    eval_error("resolve-module: expected module name");
  }
  if (!is_pair(name->data)) {
    eval_error("resolve-module: module name must be a list");
  }
  return scm_resolve_module(cast<SCM_List>(name->data));
}

SCM *scm_c_module_ref(SCM *module, SCM *symbol, SCM *default_val) {
  if (!is_module(module)) {
    eval_error("module-ref: expected module");
  }
  if (!is_sym(symbol)) {
    eval_error("module-ref: expected symbol");
  }
  
  SCM_Module *mod = cast<SCM_Module>(module);
  SCM_Symbol *sym = cast<SCM_Symbol>(symbol);
  SCM *var = scm_module_variable(mod, sym, false);
  
  if (var && !is_falsy(var)) {
    // Return variable's value (simplified implementation, assume var is the value)
    return var;
  }
  
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
  
  // Create module
  SCM *module_scm = scm_make_module(name, 31);
  
  // Process options (simplified handling, need to parse #:use-module, #:export, etc.)
  // TODO: Implement complete option parsing
  
  // Register module
  if (!g_module_registry) {
    g_module_registry = new SCM_HashTable();
    g_module_registry->capacity = 61;
    g_module_registry->size = 0;
    g_module_registry->buckets = (SCM **)calloc(61, sizeof(SCM *));
    for (size_t i = 0; i < 61; i++) {
      g_module_registry->buckets[i] = scm_nil();
    }
  }
  scm_c_hash_set_eq(wrap(g_module_registry), wrap(name), module_scm);
  
  // Set current module
  scm_set_current_module(module_scm);
  
  return module_scm;
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
  scm_c_hash_set_eq(wrap(g_module_registry), wrap(name), g_root_module);
  
  // Register built-in functions
  scm_define_function("current-module", 0, 0, 0, scm_c_current_module);
  scm_define_function("set-current-module", 1, 0, 0, scm_c_set_current_module);
  scm_define_function("resolve-module", 1, 0, 0, scm_c_resolve_module);
  scm_define_function("module-ref", 2, 1, 0, scm_c_module_ref);
  scm_define_function("module-bound?", 2, 0, 0, scm_c_module_bound_p);
}

