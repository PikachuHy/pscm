#include "pscm.h"
#include "eval.h"

// Helper function to count list length
static int count_list_length(SCM_List *l) {
  int count = 0;
  while (l) {
    count++;
    l = l->next;
  }
  return count;
}

// Helper function to check if a list node is exhausted (nil or null)
static inline bool is_list_exhausted(SCM_List *list) {
  return list == nullptr || is_nil(list->data);
}

// Helper function for for-each special form
// 
// for-each applies a procedure to elements of one or more lists in order.
// Unlike map, for-each returns an unspecified value (none).
//
// Examples:
//   (for-each display '(1 2 3))           ; prints: 123
//   (for-each abs '(4 -5 6))              ; applies abs to each element
//   (for-each (lambda (x y) (display (+ x y))) '(1 2) '(3 4))  ; prints: 46
//   (for-each (lambda (x) x) '(54 0 37))  ; single argument lambda
//
// Syntax: (for-each proc list1 list2 ...)
//   - proc: procedure or function to apply
//   - list1, list2, ...: lists of arguments (must have same length)
//
SCM *eval_for_each(SCM_Environment *env, SCM_List *l) {
  assert(l->next);
  auto f = eval_with_env(env, l->next->data);
  int arg_count;
  
  // Determine argument count based on procedure/function type
  if (is_proc(f)) {
    auto proc = cast<SCM_Procedure>(f);
    arg_count = count_list_length(proc->args);
  } else if (is_func(f)) {
    auto func = cast<SCM_Function>(f);
    // Count the number of argument lists provided
    int list_count = count_list_length(l->next->next);
    // For variable-arg functions (n_args < 0), use list_count
    // Otherwise use the function's expected argument count
    arg_count = (func->n_args < 0) ? list_count : func->n_args;
  } else {
    eval_error("for-each: first argument must be a procedure or function");
    return nullptr;
  }
  
  l = l->next->next;

  // Evaluate all argument lists and store them
  // Use stack allocation with a reasonable limit to avoid heap allocation issues with continuations
  const int MAX_FOR_EACH_ARGS = 10;
  if (arg_count > MAX_FOR_EACH_ARGS) {
    eval_error("for-each: too many arguments (max %d)", MAX_FOR_EACH_ARGS);
    return nullptr;
  }
  
  SCM_List *arg_lists[MAX_FOR_EACH_ARGS];
  for (int i = 0; i < arg_count; i++) {
    if (!l) {
      eval_error("args count not match, require %d, but got %d", arg_count, i);
      return nullptr;
    }
    auto evaluated_list = eval_with_env(env, l->data);
    // Validate that evaluated result is a list
    if (!is_pair(evaluated_list) && !is_nil(evaluated_list)) {
      eval_error("for-each: argument %d must be a list", i + 1);
      return nullptr;
    }
    arg_lists[i] = is_nil(evaluated_list) ? nullptr : cast<SCM_List>(evaluated_list);
    l = l->next;
  }

  // Iterate through all lists simultaneously
  // Continue while at least one list has elements
  while (true) {
    // Check if all lists are exhausted (early exit)
    bool all_exhausted = true;
    bool any_exhausted = false;
    for (int i = 0; i < arg_count; i++) {
      if (is_list_exhausted(arg_lists[i])) {
        any_exhausted = true;
      } else {
        all_exhausted = false;
      }
    }
    
    if (all_exhausted) {
      break;  // All lists exhausted, we're done
    }
    
    // If any list is exhausted but not all, they have different lengths
    if (any_exhausted) {
      eval_error("for-each: lists must have the same length");
      return nullptr;
    }

    // Build args_dummy structure for function calls (rebuild each iteration to avoid issues with continuations)
    SCM_List args_dummy = make_list_dummy();
    args_dummy.data = f;
    auto args_tail = &args_dummy;
    
    // Build argument list with quoted arguments and advance list pointers
    for (int i = 0; i < arg_count; i++) {
      // Wrap argument in quote to avoid re-evaluation
      auto quoted_arg = scm_list2(scm_sym_quote(), arg_lists[i]->data);
      auto arg_node = make_list(quoted_arg);
      args_tail->next = arg_node;
      args_tail = arg_node;
      arg_lists[i] = arg_lists[i]->next;
    }

    // Build and evaluate call expression
    SCM call_expr;
    call_expr.type = SCM::LIST;
    call_expr.value = &args_dummy;
    call_expr.source_loc = nullptr;  // Mark as temporary to skip call stack tracking
    
    if (debug_enabled) {
      SCM_DEBUG_EVAL("for-each ");
      print_ast(f);
      printf(" ");
      // Optimize debug output: traverse args_dummy only once
      auto args_iter = args_dummy.next;
      bool first = true;
      while (args_iter) {
        if (!first) {
          printf(" ");
        }
        first = false;
        print_ast(args_iter->data);
        args_iter = args_iter->next;
      }
      printf("\n");
    }
    
    eval_with_env(env, &call_expr);
  }

  return scm_none();
}

