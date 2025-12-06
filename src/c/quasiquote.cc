#include "pscm.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

// Helper function to check if an expression is constant (doesn't need evaluation)
static bool is_constant_expr(SCM *expr) {
  if (!is_pair(expr)) {
    // Atoms are constant except symbols
    return !is_sym(expr);
  }
  
  // Check if it's a quote expression
  SCM_List *l = cast<SCM_List>(expr);
  if (l->data && is_sym(l->data)) {
    SCM_Symbol *sym = cast<SCM_Symbol>(l->data);
    if (strcmp(sym->data, "quote") == 0) {
      return true;
    }
  }
  return false;
}

// Helper function to combine skeletons in quasiquote expansion
static SCM *combine_skeletons(SCM_Environment *env, SCM *left, SCM *right, SCM *original) {
  // If both are constants, evaluate and quote
  if (is_constant_expr(left) && is_constant_expr(right)) {
    SCM *left_val = eval_with_env(env, left);
    SCM *right_val = eval_with_env(env, right);
    // For now, just use cons - optimization can be added later
    return scm_list3(create_sym("cons", 4), left, right);
  }
  
  // If right is nil, use (list left)
  if (is_nil(right)) {
    return scm_list2(create_sym("list", 4), left);
  }
  
  // If right is (list ...), merge left into it
  if (is_pair(right)) {
    SCM_List *right_list = cast<SCM_List>(right);
    if (right_list->data && is_sym(right_list->data)) {
      SCM_Symbol *sym = cast<SCM_Symbol>(right_list->data);
      if (strcmp(sym->data, "list") == 0) {
        // Merge: (list left ...rest)
        SCM_List *new_list = make_list(create_sym("list", 4));
        new_list->next = make_list(left);
        SCM_List *tail = new_list->next;
        SCM_List *rest = right_list->next;
        while (rest) {
          tail->next = make_list(rest->data);
          tail = tail->next;
          rest = rest->next;
        }
        return wrap(new_list);
      }
    }
  }
  
  // Default: use (cons left right)
  return scm_list3(create_sym("cons", 4), left, right);
}

// Helper function to expand quasiquote expression
static SCM *expand_quasiquote(SCM_Environment *env, SCM *expr, int nesting) {
  // Handle non-pair expressions
  if (!is_pair(expr)) {
    if (is_constant_expr(expr)) {
      return expr;  // Constants are returned as-is
    }
    // Symbols need to be quoted
    return scm_list2(scm_sym_quote(), expr);
  }
  
  SCM_List *l = cast<SCM_List>(expr);
  if (!l->data) {
    return scm_nil();
  }
  
  // Handle (unquote expr) - when nesting is 0, evaluate expr
  if (is_sym(l->data)) {
    SCM_Symbol *sym = cast<SCM_Symbol>(l->data);
    if (strcmp(sym->data, "unquote") == 0 && l->next && !l->next->next) {
      if (nesting == 0) {
        // Evaluate the unquoted expression
        return eval_with_env(env, l->next->data);
      } else {
        // Nested: reduce nesting and continue
        SCM *new_right = expand_quasiquote(env, l->next->data, nesting - 1);
        return combine_skeletons(env, scm_list2(scm_sym_quote(), scm_sym_unquote()), new_right, expr);
      }
    }
    
    // Handle (quasiquote expr) - increase nesting
    if (strcmp(sym->data, "quasiquote") == 0 && l->next && !l->next->next) {
      SCM *new_right = expand_quasiquote(env, l->next->data, nesting + 1);
      return combine_skeletons(env, scm_list2(scm_sym_quote(), scm_sym_quasiquote()), new_right, expr);
    }
    
    // Handle (unquote-splicing expr) - when nesting is 0, use append
    // Note: unquote-splicing should only appear in list context, not as a top-level form
    // This case handles (unquote-splicing expr) as a list element
    if (strcmp(sym->data, "unquote-splicing") == 0 && l->next && !l->next->next) {
      if (nesting == 0) {
        // This shouldn't happen in normal quasiquote - unquote-splicing is handled in list context
        // But if it does, treat it like unquote
        return eval_with_env(env, l->next->data);
      } else {
        SCM *new_right = expand_quasiquote(env, l->next->data, nesting - 1);
        return combine_skeletons(env, scm_list2(scm_sym_quote(), scm_sym_unquote_splicing()), new_right, expr);
      }
    }
  }
  
  // Handle unquote-splicing in list context: ((unquote-splicing expr) ...)
  // This is the normal case for ,@ in a quasiquoted list
  if (is_pair(l->data)) {
    SCM_List *first = cast<SCM_List>(l->data);
    if (first->data && is_sym(first->data)) {
      SCM_Symbol *first_sym = cast<SCM_Symbol>(first->data);
      if (strcmp(first_sym->data, "unquote-splicing") == 0 && first->next && !first->next->next) {
        if (nesting == 0) {
          // Evaluate the spliced expression
          SCM *spliced_val = eval_with_env(env, first->next->data);
          // Expand the rest of the list
          SCM *new_right = l->next ? expand_quasiquote(env, wrap(l->next), nesting) : scm_nil();
          // Use append to concatenate
          if (is_nil(new_right)) {
            // If right is nil, just return the spliced value
            return spliced_val;
          }
          // Build (append spliced_val new_right)
          return scm_list3(create_sym("append", 6), spliced_val, new_right);
        } else {
          // Nested: preserve the structure
          SCM *new_left = expand_quasiquote(env, l->data, nesting - 1);
          SCM *new_right = expand_quasiquote(env, l->next ? wrap(l->next) : scm_nil(), nesting);
          return combine_skeletons(env, new_left, new_right, expr);
        }
      }
    }
  }
  
  // Recursively expand car and cdr
  SCM *new_left = expand_quasiquote(env, l->data, nesting);
  SCM *new_right = expand_quasiquote(env, l->next ? wrap(l->next) : scm_nil(), nesting);
  return combine_skeletons(env, new_left, new_right, expr);
}
  
// Helper function for quasiquote special form
SCM *eval_quasiquote(SCM_Environment *env, SCM_List *l) {
  if (!l->next) {
    quasiquote_error("missing argument");
  }
  
  // Expand the quasiquote expression
  SCM *expanded = expand_quasiquote(env, l->next->data, 0);
  
  // Evaluate the expanded expression
  return eval_with_env(env, expanded);
}
  