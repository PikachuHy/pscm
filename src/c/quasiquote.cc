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

// Helper function to check if expr is (quote symbol) and return the symbol, or NULL
static SCM *get_quoted_symbol(SCM *expr) {
  if (!is_pair(expr)) {
    return NULL;
  }
  SCM_List *l = cast<SCM_List>(expr);
  if (l->data && is_sym(l->data)) {
    SCM_Symbol *sym = cast<SCM_Symbol>(l->data);
    if (strcmp(sym->data, "quote") == 0 && l->next && !l->next->next) {
      if (is_sym(l->next->data)) {
        return l->next->data;
      }
    }
  }
  return NULL;
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
  
  // Check if left is (quote list) - this happens when we have (list ,@expr)
  // When we have (list ,@items), we want to produce (list ...items...)
  // The right side will be (append items rest) or just items
  SCM *left_quoted_sym = get_quoted_symbol(left);
  if (left_quoted_sym && is_sym(left_quoted_sym)) {
    SCM_Symbol *left_sym = cast<SCM_Symbol>(left_quoted_sym);
    if (strcmp(left_sym->data, "list") == 0) {
      // left is (quote list), right should be merged into (list ...)
      // If right is (list ...), merge left's elements (none) into it
      if (is_pair(right)) {
        SCM_List *right_list = cast<SCM_List>(right);
        if (right_list->data && is_sym(right_list->data)) {
          SCM_Symbol *sym = cast<SCM_Symbol>(right_list->data);
          if (strcmp(sym->data, "list") == 0) {
            // right is already (list ...), just return it
            return right;
          }
          // If right is (append items rest), we need to handle it specially
          // For (list ,@items), we want (list ...items...)
          // If right is (append items nil), convert to (apply list items)
          // If right is (append items rest), convert to (append (list) (append items rest))
          // or use apply: (apply list (append items rest))
          if (strcmp(sym->data, "append") == 0) {
            // right is (append items rest)
            SCM_List *rest = right_list->next;
            if (rest && rest->data) {
              SCM *items = rest->data;
              SCM *rest_expr = rest->next ? wrap(rest->next) : scm_nil();
              if (is_nil(rest_expr)) {
                // (append items nil) -> use (apply list items)
                // This will properly construct (list ...items...)
                return scm_list3(create_sym("apply", 5), 
                               create_sym("list", 4), 
                               items);
              } else {
                // (append items rest) -> use (apply list (append items rest))
                return scm_list3(create_sym("apply", 5), 
                               create_sym("list", 4), 
                               right);
              }
            } else {
              // Malformed append, shouldn't happen - use apply list for consistency
              return scm_list3(create_sym("apply", 5),
                             create_sym("list", 4),
                             right);
            }
          }
        }
      }
      // Case 3: Otherwise, right represents a single list of arguments or a single argument.
      // Check if right is a list of symbols (from a quoted list like '(c d))
      // If so, construct the list directly using cons to preserve symbols as literals
      if (is_pair(right)) {
        SCM_List *right_list = cast<SCM_List>(right);
        // Check if this is a list containing only symbols (likely from a quoted list)
        bool all_symbols = true;
        int symbol_count = 0;
        SCM_List *check = right_list;
        while (check && all_symbols) {
          if (!is_sym(check->data)) {
            all_symbols = false;
          } else {
            symbol_count++;
          }
          check = check->next;
        }
        
        if (all_symbols && right_list && symbol_count > 0) {
          // Build (list 'list 'elem1 'elem2 ...) to produce (list elem1 elem2 ...)
          // This creates a list structure containing the symbol 'list' and the elements
          SCM_List dummy = make_list_dummy();
          SCM_List *tail = &dummy;
          
          // Start with 'list' (the list function symbol)
          tail->next = make_list(create_sym("list", 4));
          tail = tail->next;
          
          // Add 'list' symbol quoted (so it becomes a literal symbol, not a function call)
          SCM *quoted_list_sym = scm_list2(scm_sym_quote(), create_sym("list", 4));
          tail->next = make_list(quoted_list_sym);
          tail = tail->next;
          
          // Add each element quoted, in order
          SCM_List *current = right_list;
          while (current) {
            SCM *quoted_elem = scm_list2(scm_sym_quote(), current->data);
            tail->next = make_list(quoted_elem);
            tail = tail->next;
            current = current->next;
          }
          
          return wrap(dummy.next);
        }
      }
      
      // Default: use (apply list right) for other cases
      return scm_list3(create_sym("apply", 5),
                      create_sym("list", 4),
                      right);
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
        SCM *eval_result = eval_with_env(env, l->next->data);
        // If the result is a symbol, quote it to preserve it as a literal
        // This prevents symbols from being looked up as variables in the final result
        if (is_sym(eval_result)) {
          return scm_list2(scm_sym_quote(), eval_result);
        }
        return eval_result;
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
  