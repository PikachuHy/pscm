#include "pscm.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Forward declaration
SCM *scm_c_list_to_vector(SCM *list);

// Error handling helper for quasiquote
[[noreturn]] static void quasiquote_error(const char *format, ...) {
  va_list args;
  va_start(args, format);
  fprintf(stderr, "quasiquote error: ");
  vfprintf(stderr, format, args);
  fprintf(stderr, "\n");
  va_end(args);
  exit(1);
}

// Helper function to check if expr is (unquote ...)
static bool is_unquote_form(SCM *expr) {
  if (!is_pair(expr)) {
    return false;
  }
  SCM_List *l = cast<SCM_List>(expr);
  if (l->data && is_sym(l->data)) {
    SCM_Symbol *sym = cast<SCM_Symbol>(l->data);
    return strcmp(sym->data, "unquote") == 0;
  }
  return false;
}

// Helper function to check if expr is (unquote-splicing ...)
static bool is_unquote_splicing_form(SCM *expr) {
  if (!is_pair(expr)) {
    return false;
  }
  SCM_List *l = cast<SCM_List>(expr);
  if (l->data && is_sym(l->data)) {
    SCM_Symbol *sym = cast<SCM_Symbol>(l->data);
    return strcmp(sym->data, "unquote-splicing") == 0;
  }
  return false;
}

// Helper function to check if expr is (quasiquote ...)
static bool is_quasiquote_form(SCM *expr) {
  if (!is_pair(expr)) {
    return false;
  }
  SCM_List *l = cast<SCM_List>(expr);
  if (l->data && is_sym(l->data)) {
    SCM_Symbol *sym = cast<SCM_Symbol>(l->data);
    return strcmp(sym->data, "quasiquote") == 0;
  }
  return false;
}

// Helper function to get the argument from (unquote expr) or similar
static SCM *get_form_arg(SCM *form) {
  if (!is_pair(form)) {
    return nullptr;
  }
  SCM_List *l = cast<SCM_List>(form);
  if (l->next) {
    return l->next->data;
  }
  return nullptr;
}

// Helper function to get the rest of a list after the first element
static SCM *get_list_rest(SCM *list_expr) {
  if (!is_pair(list_expr)) {
    return scm_nil();
  }
  SCM_List *l = cast<SCM_List>(list_expr);
  if (l->next) {
    return wrap(l->next);
  }
  return scm_nil();
}

// Helper function to append two lists (for unquote-splicing)
// This is a simplified version that merges two lists directly
static SCM *append_two_lists(SCM *list1, SCM *list2) {
  if (is_nil(list1)) {
    return list2;
  }
  if (is_nil(list2)) {
    return list1;
  }
  
  // Copy list1
  SCM_List *l1 = cast<SCM_List>(list1);
  SCM_List dummy = make_list_dummy();
  SCM_List *tail = &dummy;
  
  // Copy all elements from list1
  SCM_List *current = l1;
  while (current) {
    tail->next = make_list(current->data);
    tail = tail->next;
    
    // Check if this is the last element (nil or dotted pair)
    if (!current->next || is_nil(wrap(current->next))) {
      break;
    }
    // Check if this is a dotted pair
    if (current->is_dotted) {
      // This shouldn't happen in a proper list, but handle it
      break;
    }
    current = current->next;
  }
  
  // Check if list1 ends with a dotted pair
  bool list1_is_dotted = false;
  SCM *list1_cdr = nullptr;
  if (l1) {
    SCM_List *last = l1;
    while (last->next && !is_nil(wrap(last->next))) {
      if (last->is_dotted) {
        list1_is_dotted = true;
        list1_cdr = last->data;
        break;
      }
      last = last->next;
    }
    if (!list1_is_dotted && last->is_dotted) {
      list1_is_dotted = true;
      list1_cdr = last->data;
    }
  }
  
  // Append list2
  if (is_pair(list2)) {
    SCM_List *l2 = cast<SCM_List>(list2);
    SCM_List *current2 = l2;
    
    while (current2) {
      tail->next = make_list(current2->data);
      tail = tail->next;
      
      // Check if this is the last element
      if (!current2->next || is_nil(wrap(current2->next))) {
        break;
      }
      // Check if this is a dotted pair
      if (current2->is_dotted) {
        tail->is_dotted = true;
        break;
      }
      current2 = current2->next;
    }
    
    // Check if list2 ends with a dotted pair
    if (l2) {
      SCM_List *last2 = l2;
      while (last2->next && !is_nil(wrap(last2->next))) {
        if (last2->is_dotted) {
          tail->is_dotted = true;
          break;
        }
        last2 = last2->next;
      }
      if (last2->is_dotted) {
        tail->is_dotted = true;
      }
    }
  } else if (!is_nil(list2)) {
    // list2 is not a list, create a dotted pair
    tail->next = make_list(list2);
    tail->next->is_dotted = true;
  }
  
  return dummy.next ? wrap(dummy.next) : scm_nil();
}

// Main quasi function: expand quasiquote expression
// Based on Guile 1.8's iqq function - direct evaluation approach
static SCM *quasi(SCM_Environment *env, SCM *p, int depth) {
  // 1. Handle (unquote ...)
  if (is_unquote_form(p)) {
    SCM *arg = get_form_arg(p);
    if (!arg) {
      quasiquote_error("unquote requires an argument");
    }
    if (depth == 1) {
      // Direct evaluation and return
      return eval_with_env(env, arg);
    } else {
      // Nested: recursively process argument with reduced depth
      SCM *expanded_arg = quasi(env, arg, depth - 1);
      return scm_list2(scm_sym_unquote(), expanded_arg);
    }
  }
  
  // 2. Handle ((unquote-splicing ...) . rest)
  if (is_pair(p)) {
    SCM_List *p_list = cast<SCM_List>(p);
    if (is_pair(p_list->data) && is_unquote_splicing_form(p_list->data)) {
      SCM *arg = get_form_arg(p_list->data);
      if (!arg) {
        quasiquote_error("unquote-splicing requires an argument");
      }
      
      // Check if rest is a dotted pair
      // In a list like (a b ,@(list ...) . c), the unquote-splicing is followed by a dotted pair
      // We need to check if p_list->next exists and if it's marked as dotted
      bool rest_is_dotted = false;
      SCM *rest_cdr_data = nullptr;
      if (p_list->next) {
        // Check if the next node is a dotted pair (i.e., it's the cdr of a dotted pair)
        if (p_list->next->is_dotted) {
          rest_is_dotted = true;
          rest_cdr_data = p_list->next->data;
        }
      }
      
      if (depth == 1) {
        // Direct evaluation and merge
        SCM *list = eval_with_env(env, arg);
        // Check if result is a list
        if (!is_pair(list) && !is_nil(list)) {
          quasiquote_error("unquote-splicing requires a list");
        }
        
        // If rest is a dotted pair, handle it specially
        if (rest_is_dotted && rest_cdr_data) {
          // Process the cdr of the dotted pair
          SCM *expanded_cdr = quasi(env, rest_cdr_data, depth);
          // Append list elements, then create dotted pair with expanded_cdr
          if (is_nil(list)) {
            // If list is empty, return just the cdr
            return expanded_cdr;
          }
          // Otherwise, copy all elements from list, then create dotted pair with expanded_cdr
          SCM_List *l = cast<SCM_List>(list);
          SCM_List dummy = make_list_dummy();
          SCM_List *tail = &dummy;
          
          // Copy all elements from list using the exact same logic as append_two_lists
          SCM_List *current = l;
          while (current) {
            tail->next = make_list(current->data);
            tail = tail->next;
            
            // Check if this is the last element (nil or dotted pair)
            // Use the exact same check as append_two_lists
            if (!current->next || is_nil(wrap(current->next))) {
              break;
            }
            // Check if this is a dotted pair
            if (current->is_dotted) {
              // This shouldn't happen in a proper list, but handle it
              break;
            }
            current = current->next;
          }
          
          // Make the last element a dotted pair with expanded_cdr
          // This creates: (elem1 elem2 ... . expanded_cdr)
          tail->is_dotted = true;
          tail->data = expanded_cdr;
          
          return dummy.next ? wrap(dummy.next) : expanded_cdr;
        } else {
          // Normal case: process rest and merge
          SCM *rest = get_list_rest(p);
          SCM *expanded_rest = quasi(env, rest, depth);
          return append_two_lists(list, expanded_rest);
        }
      } else {
        // Nested: recursively process both car and cdr
        SCM *expanded_car = quasi(env, p_list->data, depth - 1);
        if (rest_is_dotted && rest_cdr_data) {
          SCM *expanded_cdr = quasi(env, rest_cdr_data, depth);
          return scm_list2(expanded_car, expanded_cdr);
        } else {
          SCM *rest = get_list_rest(p);
          SCM *expanded_cdr = quasi(env, rest, depth);
          return scm_cons(expanded_car, expanded_cdr);
        }
      }
    }
  }
  
  // 3. Handle (quasiquote ...)
  if (is_quasiquote_form(p)) {
    SCM *arg = get_form_arg(p);
    if (!arg) {
      quasiquote_error("quasiquote requires an argument");
    }
    // Increase depth
    SCM *expanded_arg = quasi(env, arg, depth + 1);
    return scm_list2(scm_sym_quasiquote(), expanded_arg);
  }
  
  // 4. Handle (p . q) - pair
  if (is_pair(p)) {
    SCM_List *p_list = cast<SCM_List>(p);
    SCM *car_val = p_list->data;
    SCM *cdr_val = get_list_rest(p);
    
    // Check if this is a dotted pair
    bool is_dotted = false;
    SCM *cdr_data = nullptr;
    if (p_list->next && p_list->next->is_dotted) {
      is_dotted = true;
      cdr_data = p_list->next->data;
    }
    
    if (is_dotted) {
      // Dotted pair: use cons
      SCM *expanded_car = quasi(env, car_val, depth);
      SCM *expanded_cdr = quasi(env, cdr_data, depth);
      return scm_cons(expanded_car, expanded_cdr);
    } else {
      // Proper list: recursively process
      SCM *expanded_car = quasi(env, car_val, depth);
      SCM *expanded_cdr = quasi(env, cdr_val, depth);
      return scm_cons(expanded_car, expanded_cdr);
    }
  }
  
  // 5. Handle vectors: #(x ...)
  if (is_vector(p)) {
    SCM_Vector *vec = cast<SCM_Vector>(p);
    // Convert vector to list and process
    SCM_List dummy = make_list_dummy();
    SCM_List *tail = &dummy;
    for (size_t i = 0; i < vec->length; i++) {
      tail->next = make_list(vec->elements[i]);
      tail = tail->next;
    }
    SCM *list_expr = dummy.next ? wrap(dummy.next) : scm_nil();
    SCM *expanded_list = quasi(env, list_expr, depth);
    // Convert back to vector
    return scm_c_list_to_vector(expanded_list);
  }
  
  // 6. Otherwise: return directly (atoms are returned as-is)
  return p;
}

// Helper function for quasiquote special form
SCM *eval_quasiquote(SCM_Environment *env, SCM_List *l) {
  if (!l->next) {
    quasiquote_error("missing argument");
  }
  
  // Start from depth=1 (because we're already inside quasiquote)
  return quasi(env, l->next->data, 1);
}
