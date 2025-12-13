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
  
  // Check if left is (quote symbol) - but we only want to handle special cases
  // Don't treat (quote list) specially unless we're in a specific context
  // where we know it's a list constructor (like after unquote-splicing)
  // For regular quasiquote expansion, (quote list) should just be treated as
  // a quoted symbol like any other
  SCM *left_quoted_sym = get_quoted_symbol(left);
  if (left_quoted_sym && is_sym(left_quoted_sym)) {
    SCM_Symbol *left_sym = cast<SCM_Symbol>(left_quoted_sym);
    // Only handle this special case if right starts with (append ...)
    // This indicates we came from unquote-splicing expansion
    if (strcmp(left_sym->data, "list") == 0 && is_pair(right)) {
      SCM_List *right_list = cast<SCM_List>(right);
      if (right_list->data && is_sym(right_list->data)) {
        SCM_Symbol *sym = cast<SCM_Symbol>(right_list->data);
        // Only merge if right is (append ...) - this means we're in unquote-splicing context
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
  
  // Special case: if this list node itself is marked as dotted pair's cdr
  // This happens when we wrap a dotted pair node: the node itself is the wrapper
  // with is_dotted=true and data=cdr. We need to handle this case specially.
  // When we call expand_quasiquote(env, wrap(l->next), nesting) where l->next is
  // a dotted pair node, the wrapped l->next becomes the root, so l->is_dotted == true
  // and l->data is the cdr. But we don't have access to the car in this context.
  // So we can't handle this case here - it should be handled by the caller.
  // However, is_dotted is on the next node, not the current node.
  // So we check l->next->is_dotted in the normal flow below.
  
  // Handle (unquote expr) - when nesting is 0, evaluate expr
  if (is_sym(l->data)) {
    SCM_Symbol *sym = cast<SCM_Symbol>(l->data);
    if (strcmp(sym->data, "unquote") == 0 && l->next && !l->next->next) {
      if (nesting == 0) {
        // Evaluate the unquoted expression
        SCM *eval_result = eval_with_env(env, l->next->data);
        // In Scheme quasiquote, unquote inserts the value directly.
        // However, when building the expanded expression, we need to quote symbols and lists
        // to prevent them from being evaluated as variables or function calls.
        // The combine_skeletons function will handle these quoted values correctly
        // when building the final (list ...) or (cons ...) structure.
        if (is_sym(eval_result) || is_pair(eval_result)) {
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
          // Check if the rest of the list is a dotted pair
          // l->next is the rest of the list after (unquote-splicing ...)
          // If l->next is a dotted pair node, then rest_is_dotted is true
          // If l->next is a list and l->next->next is a dotted pair node, we also need to check that
          bool rest_is_dotted = false;
          SCM *rest_cdr = nullptr;
          if (l->next && l->next->is_dotted) {
            // l->next is directly a dotted pair node
            rest_is_dotted = true;
            // The cdr of the dotted pair is in l->next->data
            rest_cdr = l->next->data;
          } else if (l->next && l->next->next && l->next->next->is_dotted) {
            // l->next is a list, and l->next->next is a dotted pair node
            rest_is_dotted = true;
            // The cdr of the dotted pair is in l->next->next->data
            rest_cdr = l->next->next->data;
          }
          
          // Expand the rest of the list (which may be a dotted pair)
          SCM *new_right;
          if (rest_is_dotted) {
            // Special case: l->next is a dotted pair node
            // We need to expand the cdr separately and create a dotted pair expression
            SCM *new_cdr = expand_quasiquote(env, rest_cdr, nesting);
            
            // Create a dotted pair expression: (cons ... new_cdr)
            // The car will be the result of appending spliced_val
            // If spliced_val is empty, we should directly return new_cdr (the dotted pair expression)
            // because append(() list) = list, and we want to avoid creating (() . ...)
            if (is_nil(spliced_val)) {
              // Empty splice: directly return the dotted pair expression
              // This avoids creating (() . ...) when the splice is empty
              return new_cdr;
            } else {
              // Non-empty splice: use append to concatenate spliced_val with dotted pair
              SCM *quoted_spliced = scm_list2(scm_sym_quote(), spliced_val);
              // Create (cons '() new_cdr) for the dotted pair part
              SCM *dotted_pair = scm_list3(create_sym("cons", 4), scm_list2(scm_sym_quote(), scm_nil()), new_cdr);
              new_right = scm_list3(create_sym("append", 6), quoted_spliced, dotted_pair);
              return new_right;
            }
          } else {
            new_right = l->next ? expand_quasiquote(env, wrap(l->next), nesting) : scm_nil();
          }
          
          // Use append to concatenate
          if (is_nil(new_right)) {
            // If right is nil, just return the spliced value
            // But we need to quote it to prevent re-evaluation when eval_quasiquote evaluates the result
            return scm_list2(scm_sym_quote(), spliced_val);
          }
          // Build (append spliced_val new_right)
          // Note: spliced_val is already evaluated, so we need to quote it to prevent re-evaluation
          // when append is called. However, append expects a list value, not an expression.
          // The issue is that when (append spliced_val new_right) is evaluated,
          // eval_list_with_env will try to evaluate spliced_val again, which would fail
          // if spliced_val is a list like (4 5 6) because it would try to call 4 as a function.
          // Solution: quote spliced_val so it's treated as a literal value
          SCM *quoted_spliced = scm_list2(scm_sym_quote(), spliced_val);
          
          // Special handling: if spliced_val is empty and new_right is a dotted pair expression,
          // append should return new_right directly (since append(() list) = list)
          // But we can't check this here because new_right is an expression, not a value
          // So we rely on append to handle this correctly
          return scm_list3(create_sym("append", 6), quoted_spliced, new_right);
        } else {
          // Nested: preserve the structure
          SCM *new_left = expand_quasiquote(env, l->data, nesting - 1);
          SCM *new_right = expand_quasiquote(env, l->next ? wrap(l->next) : scm_nil(), nesting);
          return combine_skeletons(env, new_left, new_right, expr);
        }
      }
    }
  }
  
  // Check if this is a dotted pair
  bool is_dotted = false;
  SCM *cdr_val = nullptr;
  if (l->next && l->next->is_dotted) {
    is_dotted = true;
    cdr_val = l->next->data;
  }
  
  // Recursively expand car and cdr
  SCM *new_left = expand_quasiquote(env, l->data, nesting);
  if (is_dotted) {
    // This is a dotted pair: expand the cdr separately
    SCM *new_cdr = expand_quasiquote(env, cdr_val, nesting);
    // Use (cons new_left new_cdr) to create the dotted pair
    // This will evaluate to a dotted pair when the cdr is not a list
    return scm_list3(create_sym("cons", 4), new_left, new_cdr);
  } else {
    // Regular list: expand cdr recursively
    SCM *new_right = expand_quasiquote(env, l->next ? wrap(l->next) : scm_nil(), nesting);
    return combine_skeletons(env, new_left, new_right, expr);
  }
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
  