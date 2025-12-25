#include "pscm.h"
#include "eval.h"


// Helper function to compute content-based hash for symbols and strings
static inline unsigned long hash_string_content(const char *data, int len) {
  unsigned long h = 0;
  for (int i = 0; i < len; i++) {
    h = h * 31 + (unsigned char)data[i];
  }
  return h;
}

// Hash function for eq? (pointer equality)
static unsigned long scm_hash_code_eq(SCM *key) {
  // For symbols, use content-based hash (same as equal?) to ensure compatibility
  // This allows hash-set! and hashq-get-handle to work on the same table
  // In Guile, symbols are interned so eq? and equal? hash to the same value
  if (is_sym(key)) {
    auto sym = cast<SCM_Symbol>(key);
    return hash_string_content(sym->data, sym->len);
  }
  // For eq?, use pointer address as hash
  return (unsigned long)(uintptr_t)key;
}

// Hash function for eqv? (value equality for numbers/chars)
static unsigned long scm_hash_code_eqv(SCM *key) {
  if (is_num(key)) {
    return (unsigned long)(int64_t)key->value;
  }
  if (is_char(key)) {
    return (unsigned char)ptr_to_char(key->value);
  }
  if (is_bool(key)) {
    return is_true(key) ? 1 : 0;
  }
  // Fallback to pointer hash
  return scm_hash_code_eq(key);
}

// Hash function for equal? (deep equality)
// Helper function to compute hash code for a list recursively
static unsigned long hash_list_content(SCM *list) {
  if (!list || !is_pair(list)) {
    return 0;
  }
  auto l = cast<SCM_List>(list);
  unsigned long hash = 0;
  while (l) {
    // Hash the car
    unsigned long car_hash = 0;
    if (is_sym(l->data)) {
      auto sym = cast<SCM_Symbol>(l->data);
      car_hash = hash_string_content(sym->data, sym->len);
    } else if (is_str(l->data)) {
      auto str = cast<SCM_String>(l->data);
      car_hash = hash_string_content(str->data, str->len);
    } else if (is_num(l->data)) {
      car_hash = (unsigned long)(int64_t)l->data->value;
    } else if (is_pair(l->data)) {
      car_hash = hash_list_content(l->data);
    } else {
      car_hash = (unsigned long)(uintptr_t)l->data;
    }
    // Combine hashes
    hash = hash * 31 + car_hash;
    l = l->next;
  }
  return hash;
}

static unsigned long scm_hash_code_equal(SCM *key) {
  // For symbols, use content-based hash (not pointer-based) for equal?
  // This ensures that different symbol objects with the same name hash to the same value
  if (is_sym(key)) {
    auto sym = cast<SCM_Symbol>(key);
    return hash_string_content(sym->data, sym->len);
  }
  // For strings, use content-based hash
  if (is_str(key)) {
    auto str = cast<SCM_String>(key);
    return hash_string_content(str->data, str->len);
  }
  // For numbers, use value-based hash
  if (is_num(key)) {
    return (unsigned long)(int64_t)key->value;
  }
  // For lists, use content-based hash
  if (is_pair(key)) {
    return hash_list_content(key);
  }
  // Fallback to pointer hash
  return scm_hash_code_eq(key);
}

// Comparison functions
static bool scm_cmp_eq(SCM *lhs, SCM *rhs) {
  // For symbols, use content-based comparison (same as equal?) to ensure compatibility
  // This allows hash-set! and hashq-get-handle to work on the same table
  // In Guile, symbols are interned so eq? and equal? return the same value
  if (is_sym(lhs) && is_sym(rhs)) {
    auto sym1 = cast<SCM_Symbol>(lhs);
    auto sym2 = cast<SCM_Symbol>(rhs);
    if (sym1->len != sym2->len) {
      return false;
    }
    // Use memcmp since we already know lengths are equal
    return memcmp(sym1->data, sym2->data, sym1->len) == 0;
  }
  return lhs == rhs;
}

static bool scm_cmp_eqv(SCM *lhs, SCM *rhs) {
  if (lhs == rhs) return true;
  if (lhs->type != rhs->type) return false;
  if (is_num(lhs)) {
    return lhs->value == rhs->value;
  }
  if (is_char(lhs)) {
    return ptr_to_char(lhs->value) == ptr_to_char(rhs->value);
  }
  if (is_bool(lhs)) {
    return is_true(lhs) == is_true(rhs);
  }
  return lhs == rhs;
}

static bool scm_cmp_equal(SCM *lhs, SCM *rhs) {
  return _eq(lhs, rhs);
}

// Helper function to validate hash table and compute bucket index
// Returns the hash table pointer, or nullptr if invalid
static SCM_HashTable *validate_and_get_bucket_idx(SCM *table, SCM *key,
                                                   unsigned long (*hash_func)(SCM *),
                                                   size_t *idx_out, const char *func_name) {
  if (!is_hash_table(table)) {
    eval_error("%s: first argument must be a hash-table", func_name);
    return nullptr;
  }
  auto hash_table = cast<SCM_HashTable>(table);
  unsigned long hash = hash_func(key);
  *idx_out = hash % hash_table->capacity;
  return hash_table;
}

// Helper function to update existing entry's value
static void update_entry_value(SCM_List *entry, SCM *value) {
  if (entry->next) {
    entry->next->data = value;
    entry->next->is_dotted = true;  // Ensure it's marked as dotted pair
  } else {
    entry->next = make_list(value);
    entry->next->is_dotted = true;  // Mark as dotted pair's cdr
  }
}

// Helper function to find entry in bucket
static SCM_List *find_entry_in_bucket(SCM *bucket, SCM *key, bool (*cmp_func)(SCM *, SCM *)) {
  if (!bucket || is_nil(bucket)) {
    return nullptr;
  }
  auto l = cast<SCM_List>(bucket);
  while (l) {
    if (is_pair(l->data)) {
      auto entry = cast<SCM_List>(l->data);
      if (entry->data && cmp_func(key, entry->data)) {
        return entry;
      }
    }
    l = l->next;
  }
  return nullptr;
}

// Helper function to insert entry into bucket at head
// Returns the entry pair (key . value)
// Note: We must create a proper dotted pair (key . value) even when value is a list.
// Using scm_cons would expand value if it's a list, which breaks hash table semantics.
static SCM *insert_entry_to_bucket(SCM_HashTable *hash_table, size_t idx, 
                                    SCM *key, SCM *value) {
  // Create a proper dotted pair (key . value) structure
  // This ensures value is stored correctly even when it's a list
  SCM_List *entry_pair = make_list(key);
  if (is_nil(value)) {
    // For nil values, create a node containing nil so hash-ref can find it
    entry_pair->next = make_list(value);
    entry_pair->next->is_dotted = true;
  } else {
    // For non-nil values, wrap in a dotted pair node
    entry_pair->next = make_list(value);
    entry_pair->next->is_dotted = true;  // Mark as dotted pair's cdr
  }
  
  auto entry_node = make_list(wrap(entry_pair));
  if (hash_table->buckets[idx] && !is_nil(hash_table->buckets[idx])) {
    entry_node->next = cast<SCM_List>(hash_table->buckets[idx]);
  } else {
    entry_node->next = nullptr;
  }
  hash_table->buckets[idx] = wrap(entry_node);
  hash_table->size++;
  return wrap(entry_pair);
}

// Helper function to remove entry from bucket
// Returns true if entry was found and removed
static bool remove_entry_from_bucket(SCM **bucket_ptr, SCM *key, bool (*cmp_func)(SCM *, SCM *)) {
  if (!bucket_ptr || !*bucket_ptr || is_nil(*bucket_ptr)) {
    return false;
  }
  auto l = cast<SCM_List>(*bucket_ptr);
  SCM_List *prev = nullptr;
  while (l) {
    if (is_pair(l->data)) {
      auto entry = cast<SCM_List>(l->data);
      if (entry->data && cmp_func(key, entry->data)) {
        // Found the entry, remove it
        if (prev) {
          prev->next = l->next;
        } else {
          // First entry in bucket
          if (l->next) {
            *bucket_ptr = wrap(l->next);
          } else {
            *bucket_ptr = scm_nil();
          }
        }
        return true;
      }
    }
    prev = l;
    l = l->next;
  }
  return false;
}

// make-hash-table: Create a new hash table
SCM *scm_c_make_hash_table(SCM *size_arg) {
  size_t capacity = 31;  // Default capacity
  if (size_arg && !is_nil(size_arg)) {
    if (!is_num(size_arg)) {
      eval_error("make-hash-table: size must be an integer");
      return nullptr;
    }
    capacity = (size_t)(int64_t)size_arg->value;
    if (capacity == 0) {
      capacity = 31;
    }
  }
  
  auto hash_table = new SCM_HashTable();
  hash_table->capacity = capacity;
  hash_table->size = 0;
  hash_table->buckets = (SCM **)calloc(capacity, sizeof(SCM *));
  // Initialize all buckets to nil
  // Note: calloc initializes to 0, but we need scm_nil() which may not be 0
  for (size_t i = 0; i < capacity; i++) {
    hash_table->buckets[i] = scm_nil();
  }
  
  return wrap(hash_table);
}

// hash-set!: Set a key-value pair in hash table
SCM *scm_c_hash_set(SCM *table, SCM *key, SCM *value, 
                    unsigned long (*hash_func)(SCM *),
                    bool (*cmp_func)(SCM *, SCM *)) {
  size_t idx;
  auto hash_table = validate_and_get_bucket_idx(table, key, hash_func, &idx, "hash-set!");
  if (!hash_table) {
    return nullptr;
  }
  
  // Check if key already exists
  auto entry = find_entry_in_bucket(hash_table->buckets[idx], key, cmp_func);
  if (entry) {
    update_entry_value(entry, value);
    return value;
  }
  
  // Create new entry and insert at head of bucket
  insert_entry_to_bucket(hash_table, idx, key, value);
  return value;
}

// hash-ref: Get value from hash table
SCM *scm_c_hash_ref(SCM *table, SCM *key,
                    unsigned long (*hash_func)(SCM *),
                    bool (*cmp_func)(SCM *, SCM *)) {
  size_t idx;
  auto hash_table = validate_and_get_bucket_idx(table, key, hash_func, &idx, "hash-ref");
  if (!hash_table) {
    return nullptr;
  }
  
  auto entry = find_entry_in_bucket(hash_table->buckets[idx], key, cmp_func);
  if (entry) {
    // entry is (key . value) pair
    // If entry->next is nullptr, it means value is nil (from scm_cons when cdr is nil)
    // If entry->next exists, entry->next->data is the value
    if (entry->next) {
      return entry->next->data;
    } else {
      // entry->next is nullptr means value is nil
      return scm_nil();
    }
  }
  
  return scm_bool_false();
}

// hash-get-handle: Get handle (key . value) pair from hash table
SCM *scm_c_hash_get_handle(SCM *table, SCM *key,
                           unsigned long (*hash_func)(SCM *),
                           bool (*cmp_func)(SCM *, SCM *)) {
  size_t idx;
  auto hash_table = validate_and_get_bucket_idx(table, key, hash_func, &idx, "hash-get-handle");
  if (!hash_table) {
    return nullptr;
  }
  
  auto entry = find_entry_in_bucket(hash_table->buckets[idx], key, cmp_func);
  if (entry) {
    // entry is SCM_List* representing (key . value)
    // Return the entry pair directly (wrapped)
    return wrap(entry);
  }
  
  return scm_bool_false();
}

// hash-create-handle!: Create or get handle for key-value pair
SCM *scm_c_hash_create_handle(SCM *table, SCM *key, SCM *value,
                               unsigned long (*hash_func)(SCM *),
                               bool (*cmp_func)(SCM *, SCM *)) {
  size_t idx;
  auto hash_table = validate_and_get_bucket_idx(table, key, hash_func, &idx, "hash-create-handle!");
  if (!hash_table) {
    return nullptr;
  }
  
  // Check if key already exists
  auto entry = find_entry_in_bucket(hash_table->buckets[idx], key, cmp_func);
  if (entry) {
    update_entry_value(entry, value);
    // Return the entry pair (key . value)
    return scm_cons(entry->data, entry->next->data);
  }
  
  // Create new entry and insert at head of bucket
  return insert_entry_to_bucket(hash_table, idx, key, value);
}

// hash-remove!: Remove a key from hash table
SCM *scm_c_hash_remove(SCM *table, SCM *key,
                       unsigned long (*hash_func)(SCM *),
                       bool (*cmp_func)(SCM *, SCM *)) {
  size_t idx;
  auto hash_table = validate_and_get_bucket_idx(table, key, hash_func, &idx, "hash-remove!");
  if (!hash_table) {
    return nullptr;
  }
  
  if (remove_entry_from_bucket(&hash_table->buckets[idx], key, cmp_func)) {
    hash_table->size--;
  }
  
  return scm_none();
}

// hash-fold: Fold over hash table entries
SCM *scm_c_hash_fold(SCM *proc, SCM *init, SCM *table,
                     unsigned long (*hash_func)(SCM *),
                     bool (*cmp_func)(SCM *, SCM *)) {
  if (!is_hash_table(table)) {
    eval_error("hash-fold: third argument must be a hash-table");
    return nullptr;
  }
  
  auto hash_table = cast<SCM_HashTable>(table);
  SCM *result = init;
  
  for (size_t i = 0; i < hash_table->capacity; i++) {
    auto bucket = hash_table->buckets[i];
    if (bucket && !is_nil(bucket)) {
      auto l = cast<SCM_List>(bucket);
      while (l) {
        if (is_pair(l->data)) {
          auto entry = cast<SCM_List>(l->data);
          if (entry->data && entry->next) {
            SCM *key = entry->data;
            SCM *value = entry->next->data;
            
            // Call proc with (key value result)
            // Build call expression: (proc key value result)
            // Wrap arguments in quote to prevent re-evaluation (same as for_each.cc)
            SCM_List args_dummy = make_list_dummy();
            args_dummy.data = proc;
            auto args_tail = &args_dummy;
            // Wrap key in quote
            SCM *quoted_key = scm_list2(scm_sym_quote(), key);
            args_tail->next = make_list(quoted_key);
            args_tail = args_tail->next;
            // Wrap value in quote
            SCM *quoted_value = scm_list2(scm_sym_quote(), value);
            args_tail->next = make_list(quoted_value);
            args_tail = args_tail->next;
            // Wrap result in quote
            SCM *quoted_result = scm_list2(scm_sym_quote(), result);
            args_tail->next = make_list(quoted_result);
            
            // Build and evaluate call expression (same as for_each.cc)
            SCM call_expr;
            call_expr.type = SCM::LIST;
            call_expr.value = &args_dummy;
            call_expr.source_loc = nullptr;  // Mark as temporary to skip call stack tracking
            result = eval_with_env(&g_env, &call_expr);
          }
        }
        l = l->next;
      }
    }
  }
  
  return result;
}

// Public API functions - wrapper functions with correct signatures
SCM *scm_c_make_hash_table_wrapper(SCM_List *args) {
  if (!args || !args->data) {
    return scm_c_make_hash_table(nullptr);
  }
  return scm_c_make_hash_table(args->data);
}

SCM *scm_c_hash_set_eq(SCM *table, SCM *key, SCM *value) {
  return scm_c_hash_set(table, key, value, scm_hash_code_eq, scm_cmp_eq);
}

SCM *scm_c_hash_set_eqv(SCM *table, SCM *key, SCM *value) {
  return scm_c_hash_set(table, key, value, scm_hash_code_eqv, scm_cmp_eqv);
}

SCM *scm_c_hash_set_equal(SCM *table, SCM *key, SCM *value) {
  return scm_c_hash_set(table, key, value, scm_hash_code_equal, scm_cmp_equal);
}

SCM *scm_c_hash_ref_eq(SCM *table, SCM *key) {
  return scm_c_hash_ref(table, key, scm_hash_code_eq, scm_cmp_eq);
}

SCM *scm_c_hash_ref_eqv(SCM *table, SCM *key) {
  return scm_c_hash_ref(table, key, scm_hash_code_eqv, scm_cmp_eqv);
}

SCM *scm_c_hash_ref_equal(SCM *table, SCM *key) {
  return scm_c_hash_ref(table, key, scm_hash_code_equal, scm_cmp_equal);
}

SCM *scm_c_hash_get_handle_eq(SCM *table, SCM *key) {
  return scm_c_hash_get_handle(table, key, scm_hash_code_eq, scm_cmp_eq);
}

SCM *scm_c_hash_get_handle_eqv(SCM *table, SCM *key) {
  return scm_c_hash_get_handle(table, key, scm_hash_code_eqv, scm_cmp_eqv);
}

SCM *scm_c_hash_get_handle_equal(SCM *table, SCM *key) {
  return scm_c_hash_get_handle(table, key, scm_hash_code_equal, scm_cmp_equal);
}

SCM *scm_c_hash_create_handle_eq(SCM *table, SCM *key, SCM *value) {
  return scm_c_hash_create_handle(table, key, value, scm_hash_code_eq, scm_cmp_eq);
}

SCM *scm_c_hash_create_handle_eqv(SCM *table, SCM *key, SCM *value) {
  return scm_c_hash_create_handle(table, key, value, scm_hash_code_eqv, scm_cmp_eqv);
}

SCM *scm_c_hash_create_handle_equal(SCM *table, SCM *key, SCM *value) {
  return scm_c_hash_create_handle(table, key, value, scm_hash_code_equal, scm_cmp_equal);
}

SCM *scm_c_hash_remove_eq(SCM *table, SCM *key) {
  return scm_c_hash_remove(table, key, scm_hash_code_eq, scm_cmp_eq);
}

SCM *scm_c_hash_remove_eqv(SCM *table, SCM *key) {
  return scm_c_hash_remove(table, key, scm_hash_code_eqv, scm_cmp_eqv);
}

SCM *scm_c_hash_remove_equal(SCM *table, SCM *key) {
  return scm_c_hash_remove(table, key, scm_hash_code_equal, scm_cmp_equal);
}

SCM *scm_c_hash_fold_wrapper(SCM *proc, SCM *init, SCM *table) {
  return scm_c_hash_fold(proc, init, table, scm_hash_code_equal, scm_cmp_equal);
}

void init_hash_table() {
  // Register hash table functions
  scm_define_vararg_function("make-hash-table", scm_c_make_hash_table_wrapper);
  scm_define_function("hash-set!", 3, 0, 0, scm_c_hash_set_equal);
  scm_define_function("hashq-set!", 3, 0, 0, scm_c_hash_set_eq);
  scm_define_function("hashv-set!", 3, 0, 0, scm_c_hash_set_eqv);
  scm_define_function("hash-ref", 2, 0, 0, scm_c_hash_ref_equal);
  scm_define_function("hashq-ref", 2, 0, 0, scm_c_hash_ref_eq);
  scm_define_function("hashv-ref", 2, 0, 0, scm_c_hash_ref_eqv);
  scm_define_function("hash-get-handle", 2, 0, 0, scm_c_hash_get_handle_equal);
  scm_define_function("hashq-get-handle", 2, 0, 0, scm_c_hash_get_handle_eq);
  scm_define_function("hashv-get-handle", 2, 0, 0, scm_c_hash_get_handle_eqv);
  scm_define_function("hash-create-handle!", 3, 0, 0, scm_c_hash_create_handle_equal);
  scm_define_function("hashq-create-handle!", 3, 0, 0, scm_c_hash_create_handle_eq);
  scm_define_function("hashv-create-handle!", 3, 0, 0, scm_c_hash_create_handle_eqv);
  scm_define_function("hash-remove!", 2, 0, 0, scm_c_hash_remove_equal);
  scm_define_function("hashq-remove!", 2, 0, 0, scm_c_hash_remove_eq);
  scm_define_function("hashv-remove!", 2, 0, 0, scm_c_hash_remove_eqv);
  scm_define_function("hash-fold", 3, 0, 0, scm_c_hash_fold_wrapper);
}

