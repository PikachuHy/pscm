#include "pscm.h"
#include "eval.h"

SCM *expand_let(SCM *expr) {
  auto l = cast<SCM_List>(expr);
  assert(l->next);
  assert(l->next->next);
  SCM_List dummy_params = make_list_dummy();
  SCM_List dummy_args = make_list_dummy();
  auto params = &dummy_params;
  auto args = &dummy_args;
  auto func_args = l->next->data;
  if (is_nil(func_args)) {
    // (let () body) -> ((lambda () body))
    auto new_l = make_list(scm_sym_lambda());
    new_l->next = make_list(scm_nil());
    new_l->next->next = l->next->next;
    auto lambda_expr = wrap(new_l);
    auto call_expr = make_list(lambda_expr);
    auto ast = wrap(call_expr);
    if (debug_enabled) {
      SCM_DEBUG_EVAL("expand let () ");
      print_ast(ast);
      printf("\n");
    }
    return ast;
  }
  else if (is_sym(func_args)) {
    /*
(let loop ((arg1 val1) (arg2 val2))
  body
  (loop ...))
->
(letrec ((loop (lambda (arg1 arg2)
               body
               (loop ...))))
  (loop val1 val2))
    
    Important: The initialization expressions (val1, val2) are evaluated
    in the OUTER environment, before the letrec binding is established.
    So if there's a variable shadowing issue, we need to evaluate the
    args in the outer environment first.
    */
    assert(l->next->next->next);
    auto argl = l->next->next->data;
    auto body = l->next->next->next->data;
    
    // Handle empty bindings list for named let: (let name () body)
    SCM *params = scm_nil();
    SCM *args = scm_nil();
    
    if (!is_nil(argl)) {
      // Only map if bindings list is not empty
      if (!is_pair(argl)) {
        eval_error("let: bindings must be a list");
        return nullptr;
      }
      params = map(argl, car);
      args = map(argl, cadr);
    }
    
    if (debug_enabled) {
      SCM_ERROR_EVAL("expand named let\n");
      printf("params ");
      print_ast(params);
      printf("\n");
      printf("args ");
      print_ast(args);
      printf("\n");
      print_ast(body);
      printf("\n");
    }
    auto val = scm_list3(scm_sym_lambda(), params, body);
    auto letrec_arg = scm_list2(func_args, val);
    
    // For empty args, just call the function: (name)
    // For non-empty args, concatenate: (name arg1 arg2 ...)
    //
    // Important: In named let, the initialization expressions (args) are
    // evaluated in the OUTER environment, before the letrec binding is
    // established. This means if the named let variable shadows an outer
    // variable, the args should still reference the outer variable.
    //
    // The problem: when we expand to letrec, the args are evaluated in the
    // letrec environment where the binding already exists (initialized to #f).
    // To fix this, we need to evaluate the args in the outer environment first,
    // then pass the evaluated values to the letrec.
    //
    // The correct expansion:
    // (let ((temp1 val1) (temp2 val2))  ; evaluate args in outer env
    //   (letrec ((name (lambda (param1 param2) body)))
    //     (name temp1 temp2)))
    //
    // We need to generate temporary variable names. For simplicity, we'll
    // use a gensym-like approach, but for now let's use a simpler method:
    // wrap the letrec in a let that evaluates the args first.
    SCM *call_fn;
    if (is_nil(args)) {
      call_fn = scm_list1(func_args);
      auto new_expr = scm_list3(scm_sym_letrec(), scm_list1(letrec_arg), call_fn);
      if (debug_enabled) {
        printf("expand -> ");
        print_ast(new_expr);
        printf("\n");
      }
      return new_expr;
    } else {
      // For non-empty args, we need to evaluate them in the outer environment.
      // The problem: when we expand to letrec, the args are evaluated in the
      // letrec environment where the binding already exists (initialized to #f).
      // 
      // The correct fix: expand to use let to evaluate args first in outer env,
      // then letrec. We need to generate temporary variable names.
      //
      // Use a simple approach: generate temp names using a counter.
      // We'll use names like "__temp_0", "__temp_1", etc.
      SCM_List temp_bindings_dummy = make_list_dummy();
      SCM_List temp_params_dummy = make_list_dummy();
      auto temp_bindings = &temp_bindings_dummy;
      auto temp_params = &temp_params_dummy;
      
      auto argl_it = cast<SCM_List>(argl);
      int temp_counter = 0;
      while (argl_it) {
        auto param = car(argl_it->data);
        auto arg_expr = cadr(argl_it->data);
        
        // Generate a temporary variable name using a simple counter
        char temp_name[64];
        snprintf(temp_name, sizeof(temp_name), "__temp_%d", temp_counter++);
        SCM *temp_sym_scm = create_sym(temp_name, strlen(temp_name));
        
        // Create binding: (temp-sym arg-expr)
        auto temp_binding = scm_list2(temp_sym_scm, arg_expr);
        temp_bindings->next = make_list(temp_binding);
        temp_bindings = temp_bindings->next;
        
        // Add temp to params list for the letrec call
        temp_params->next = make_list(temp_sym_scm);
        temp_params = temp_params->next;
        
        argl_it = argl_it->next;
      }
      
      // Now create the letrec call using temp params
      // Build the function call: (name temp1 temp2 ...)
      if (temp_params_dummy.next) {
        // Build list: (name temp1 temp2 ...)
        SCM_List call_dummy = make_list_dummy();
        auto call_tail = &call_dummy;
        call_tail->next = make_list(func_args);
        call_tail = call_tail->next;
        
        // Add all temp params
        auto temp_it = temp_params_dummy.next;
        while (temp_it) {
          call_tail->next = make_list(temp_it->data);
          call_tail = call_tail->next;
          temp_it = temp_it->next;
        }
        call_fn = wrap(call_dummy.next);
      } else {
        call_fn = scm_list1(func_args);
      }
      
      // Create: (let ((__temp_0 arg1) (__temp_1 arg2))
      //          (letrec ((name (lambda (p1 p2) body)))
      //            (name __temp_0 __temp_1)))
      auto inner_letrec = scm_list3(scm_sym_letrec(), scm_list1(letrec_arg), call_fn);
      auto outer_let = scm_list3(scm_sym_let(), wrap(temp_bindings_dummy.next), inner_letrec);
      
      if (debug_enabled) {
        printf("expand -> ");
        print_ast(outer_let);
        printf("\n");
      }
      return outer_let;
    }
  }
  else {
    if (debug_enabled) {
      if (!is_pair(func_args)) {
        SCM_ERROR_EVAL("expand_let error ");
        print_ast(expr);
        printf(" func_args is ");
        print_ast(func_args);
        printf("\n");
      }
    }
    assert(is_pair(func_args));
    auto argl = cast<SCM_List>(func_args);
    while (argl) {
      assert(argl->data);
      assert(is_pair(argl->data));
      auto arg = cast<SCM_List>(argl->data);
      assert(arg->next);
      auto new_param = make_list(arg->data);
      params->next = new_param;
      params = params->next;

      auto new_arg = make_list(arg->next->data);
      args->next = new_arg;
      args = args->next;
      argl = argl->next;
    }
  }
  auto new_l = make_list(scm_sym_lambda());
  new_l->next = make_list(wrap(dummy_params.next));
  new_l->next->next = l->next->next;
  new_l = make_list(wrap(new_l));
  new_l->next = dummy_args.next;
  auto ast = wrap(new_l);
  if (debug_enabled) {
    SCM_DEBUG_EVAL("expand let ");
    print_ast(ast);
    printf("\n");
  }
  return ast;
#if 0
    SCM_Procedure *proc = new SCM_Procedure();
    proc->name = NULL;
    proc->args = dummy_params.next;
    proc->body = l->next->next;
    proc->env = env;

    auto new_list = new SCM_List();
    new_list->data = wrap(proc);
    new_list->next = dummy_args.next;
    new_list->is_dotted = false;
    auto new_l = new SCM();
    new_l->type = SCM::LIST;
    new_l->value = new_list;
    ast = new_l;
    if (debug_enabled) {
      SCM_DEBUG_EVAL("expand let ");
      printf("new expr: ");
      print_ast(ast);
      printf("\n");
      printf("func body: ");
      print_list(proc->body);
      printf("\n");
    }
#endif
}

SCM *expand_letstar(SCM *expr) {
  auto l = cast<SCM_List>(expr);
  assert(l->next);
  assert(l->next->next);
  SCM *ast = nullptr;
  auto bindings = l->next->data;
  
  // Handle empty bindings list
  if (is_nil(bindings)) {
    // (let* () body) -> ((lambda () body))
    auto new_l = make_list(scm_sym_lambda());
    new_l->next = make_list(scm_nil());
    new_l->next->next = l->next->next;
    auto lambda_expr = wrap(new_l);
    auto call_expr = make_list(lambda_expr);
    auto ast = wrap(call_expr);
    if (debug_enabled) {
      SCM_DEBUG_EVAL("expand let* () ");
      print_ast(ast);
      printf("\n");
    }
    return ast;
  }
  
  // Check if bindings is a pair
  if (!is_pair(bindings)) {
    eval_error("let*: bindings must be a list");
    return nullptr;
  }
  
  auto argl = car(bindings);
  // auto param = car(argl);
  // auto arg = cadr(argl);
  auto rest_argl = cdr(bindings);
  if (debug_enabled) {
    printf("argl ");
    print_ast(argl);
    printf("\n");
    printf("rest_argl ");
    print_ast(rest_argl);
    printf("\n");
  }
  if (is_nil(rest_argl)) {
    /*
    (let* ((arg2 ...))
          ...)
    -->
    (let ((arg2 ...))
         ...)
    */
    auto new_l = make_list(scm_sym_let());
    new_l->next = l->next;
    auto new_ast = wrap(new_l);
    ast = new_ast;
  }
  else {
    /*
    (let* ((arg1 ...)
           (arg2 ...))
          ...)
    -->
    (let ((arg1 ...))
         (let* ((arg2 ...))
               ...))
    */
    auto new_expr_arg1 = scm_list1(argl);
    auto new_letstar_arg = scm_list1(rest_argl);
    auto new_letstar_arg_l = cast<SCM_List>(new_letstar_arg);
    new_letstar_arg_l->next = l->next->next;
    auto new_l = make_list(car(expr));
    new_l->next = new_letstar_arg_l;

    auto new_let_l = make_list(scm_sym_let());
    new_let_l->next = make_list(new_expr_arg1);
    new_let_l->next->next = make_list(wrap(new_l));
    auto new_ast = wrap(new_let_l);
    ast = new_ast;
  }
  if (debug_enabled) {
    SCM_DEBUG_EVAL("expand let* ");
    print_ast(ast);
    printf("\n");
  }
  return ast;
}

// expand letrec
/*
(letrec () (define x 9) x)
expanded as
(let ()
  (define x 9) x)
*/
/*
(letrec ((fib (lambda (n)
      (cond ((zero? n) 1)
            ((= 1 n) 1)
            (else  (+ (fib (- n 1))
      (fib (- n 2))))))))
  (fib 10))
expanded as
(let ((fib #f))
  (set! fib (lambda (n)
              (cond ((zero? n) 1)
                    ((= 1 n) 1)
                    (else  (+ (fib (- n 1))
                                (fib (- n 2))))))
  (fib 10))
*/
SCM *expand_letrec(SCM *expr) {
  auto l = cast<SCM_List>(expr);
  assert(l->next);
  assert(l->next->next);
  SCM_List dummy_params = make_list_dummy();
  SCM_List dummy_args = make_list_dummy();
  auto params = &dummy_params;
  auto args = &dummy_args;
  auto func_args = l->next->data;
  if (is_nil(func_args)) {
    // (letrec () body) -> ((lambda () body))
    auto new_l = make_list(scm_sym_lambda());
    new_l->next = make_list(scm_nil());
    new_l->next->next = l->next->next;
    auto lambda_expr = wrap(new_l);
    auto call_expr = make_list(lambda_expr);
    auto ast = wrap(call_expr);
    if (debug_enabled) {
      SCM_DEBUG_EVAL("expand letrec () ");
      print_ast(ast);
      printf("\n");
    }
    return ast;
  }
  else {
    assert(is_pair(func_args));
    auto argl = cast<SCM_List>(func_args);
    while (argl) {
      assert(argl->data);
      assert(is_pair(argl->data));
      auto arg = cast<SCM_List>(argl->data);
      assert(arg->next);

      // Save arg->data and arg->next->data to avoid issues with temporary objects
      // These are SCM * pointers, so we can use them directly
      SCM *var_sym = arg->data;  // The variable name (symbol)
      SCM *var_val = arg->next->data;  // The value expression

      auto new_param = make_list(scm_list2(var_sym, scm_bool_false()));
      params->next = new_param;
      params = params->next;

      auto new_arg = make_list(scm_list3(scm_sym_set(), var_sym, var_val));
      args->next = new_arg;
      args = args->next;
      argl = argl->next;
    }
  }
  auto new_l = make_list(scm_sym_let());
  new_l->next = make_list(wrap(dummy_params.next));
  if (dummy_args.next) {
    new_l->next->next = dummy_args.next;
    // Find the last node in the args list and append the body
    // args currently points to the last node we added
    // We need to append the body (l->next->next) to the end of the args list
    SCM_List *last_args_node = args;  // args is already pointing to the last node
    // The body (l->next->next) is a SCM_List * that contains the body expressions
    // We need to copy the body list structure properly
    SCM_List *body_list = l->next->next;
    if (body_list) {
      // Body is a list of expressions, copy each element
      SCM_List *body_current = body_list;
      while (body_current) {
        SCM_List *body_node = make_list(body_current->data);
        last_args_node->next = body_node;
        last_args_node = body_node;
        body_current = body_current->next;
      }
    } else {
      // Body is empty, this shouldn't happen but handle it
      // Do nothing, last_args_node already points to the end
    }
  }
  else {
    new_l->next->next = l->next->next;
  }
  auto ast = wrap(new_l);
  if (debug_enabled) {
    SCM_DEBUG_EVAL("expand letrec ");
    print_ast(ast);
    printf("\n");
  }
  return ast;
}