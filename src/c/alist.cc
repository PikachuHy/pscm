#include "pscm.h"
#include <type_traits>
bool _eq(SCM *lhs, SCM *rhs);

// Helper function for eq? comparison (pointer/identity equality)
// This implements the eq? predicate semantics: two objects are eq? if they are the same object
static bool is_eq_pointer(SCM *lhs, SCM *rhs) {
  if (lhs == rhs) {
    return true;
  }
  if (lhs->type != rhs->type) {
    return false;
  }
  // For strings and symbols, compare pointers (value field points to the same object)
  if (is_str(lhs) || is_sym(lhs)) {
    return lhs->value == rhs->value;
  }
  // For numbers, compare values
  if (is_num(lhs)) {
    return lhs->value == rhs->value;
  }
  // For other types, use pointer comparison
  return lhs == rhs;
}

// acons: (acons key value alist) -> new alist with (key . value) prepended
SCM *scm_c_acons(SCM *key, SCM *value, SCM *alist) {
  // Create a new pair (key . value)
  SCM *pair = scm_cons(key, value);
  // Prepend to alist
  if (is_nil(alist)) {
    return scm_list1(pair);
  }
  return scm_cons(pair, alist);
}

// assoc: (assoc key alist) -> pair or #f
SCM *scm_c_assoc(SCM *key, SCM *alist) {
  if (is_nil(alist)) {
    return scm_bool_false();
  }
  auto l = cast<SCM_List>(alist);
  while (l) {
    if (is_pair(l->data)) {
      auto pair = cast<SCM_List>(l->data);
      if (pair->data && _eq(key, pair->data)) {
        return l->data; // Return the pair
      }
    }
    l = l->next;
  }
  return scm_bool_false();
}

// assoc-ref: (assoc-ref alist key) -> value or #f
SCM *scm_c_assoc_ref(SCM *alist, SCM *key) {
  SCM *pair = scm_c_assoc(key, alist);
  if (!pair || is_nil(pair) || (is_bool(pair) && is_false(pair))) {
    return scm_bool_false();
  }
  // Return the cdr of the pair
  auto pair_list = cast<SCM_List>(pair);
  if (pair_list->next) {
    return pair_list->next->data;
  }
  return scm_bool_false();
}

// assoc-set!: (assoc-set! alist key value) -> new alist
SCM *scm_c_assoc_set(SCM *alist, SCM *key, SCM *value) {
  if (is_nil(alist)) {
    // If alist is empty, create new entry
    return scm_c_acons(key, value, scm_nil());
  }
  
  auto l = cast<SCM_List>(alist);
  SCM_List dummy = make_list_dummy();
  SCM_List *tail = &dummy;
  bool found = false;
  
  while (l) {
    if (is_pair(l->data)) {
      auto pair = cast<SCM_List>(l->data);
      if (pair->data && _eq(key, pair->data)) {
        // Found matching key, replace with new pair
        SCM *new_pair = scm_cons(key, value);
        SCM_List *node = make_list(new_pair);
        tail->next = node;
        tail = node;
        found = true;
      } else {
        // Keep existing pair
        SCM_List *node = make_list(l->data);
        tail->next = node;
        tail = node;
      }
    } else {
      // Keep non-pair elements as-is
      SCM_List *node = make_list(l->data);
      tail->next = node;
      tail = node;
    }
    l = l->next;
  }
  
  if (!found) {
    // Key not found, prepend new entry
    SCM *new_pair = scm_cons(key, value);
    SCM_List *new_node = make_list(new_pair);
    new_node->next = dummy.next;
    if (new_node->next) {
      return wrap(new_node);
    }
    return scm_list1(new_pair);
  }
  
  if (dummy.next) {
    return wrap(dummy.next);
  }
  return scm_nil();
}

// assq-set!: (assq-set! alist key value) -> new alist (using eq? for comparison)
SCM *scm_c_assq_set(SCM *alist, SCM *key, SCM *value) {
  if (is_nil(alist)) {
    // If alist is empty, create new entry
    return scm_c_acons(key, value, scm_nil());
  }
  
  auto l = cast<SCM_List>(alist);
  SCM_List dummy = make_list_dummy();
  SCM_List *tail = &dummy;
  bool found = false;
  
  while (l) {
    if (is_pair(l->data)) {
      auto pair = cast<SCM_List>(l->data);
      if (pair->data && is_eq_pointer(key, pair->data)) {
        // Found matching key, replace with new pair
        SCM *new_pair = scm_cons(key, value);
        SCM_List *node = make_list(new_pair);
        tail->next = node;
        tail = node;
        found = true;
      } else {
        // Keep existing pair
        SCM_List *node = make_list(l->data);
        tail->next = node;
        tail = node;
      }
    } else {
      // Keep non-pair elements as-is
      SCM_List *node = make_list(l->data);
      tail->next = node;
      tail = node;
    }
    l = l->next;
  }
  
  if (!found) {
    // Key not found, prepend new entry
    SCM *new_pair = scm_cons(key, value);
    SCM_List *new_node = make_list(new_pair);
    new_node->next = dummy.next;
    if (new_node->next) {
      return wrap(new_node);
    }
    return scm_list1(new_pair);
  }
  
  if (dummy.next) {
    return wrap(dummy.next);
  }
  return scm_nil();
}

// assoc-remove!: (assoc-remove! alist key) -> new alist (removes only first match)
SCM *scm_c_assoc_remove(SCM *alist, SCM *key) {
  if (is_nil(alist)) {
    return scm_nil();
  }
  
  auto l = cast<SCM_List>(alist);
  SCM_List dummy = make_list_dummy();
  SCM_List *tail = &dummy;
  bool removed = false;
  
  while (l) {
    if (is_pair(l->data)) {
      auto pair = cast<SCM_List>(l->data);
      if (!removed && pair->data && _eq(key, pair->data)) {
        // Skip this pair (remove it) - only remove first match
        removed = true;
      } else {
        // Keep this pair
        SCM_List *node = make_list(l->data);
        tail->next = node;
        tail = node;
      }
    } else {
      // Keep non-pair elements
      SCM_List *node = make_list(l->data);
      tail->next = node;
      tail = node;
    }
    l = l->next;
  }
  
  if (dummy.next) {
    return wrap(dummy.next);
  }
  return scm_nil();
}

SCM *scm_c_assv(SCM *key, SCM *alist) {
  assert(key);
  assert(alist);
  assert(is_pair(alist) || is_nil(alist));
  if (is_nil(alist)) {
    return scm_bool_false();
  }
  auto l = cast<SCM_List>(alist);
  auto it = l;
  while (it) {
    if (it->data && is_pair(it->data)) {
      auto pair = cast<SCM_List>(it->data);
      if (pair->data && _eq(key, pair->data)) {
        return it->data;
      }
    }
    it = it->next;
  }
  return scm_bool_false();
}

void init_alist() {
  scm_define_function("assv", 2, 0, 0, scm_c_assv);
  scm_define_function("acons", 3, 0, 0, scm_c_acons);
  scm_define_function("assoc", 2, 0, 0, scm_c_assoc);
  scm_define_function("assoc-ref", 2, 0, 0, scm_c_assoc_ref);
  scm_define_function("assoc-set!", 3, 0, 0, scm_c_assoc_set);
  scm_define_function("assq-set!", 3, 0, 0, scm_c_assq_set);
  scm_define_function("assoc-remove!", 2, 0, 0, scm_c_assoc_remove);
}
