#include "pscm.h"

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
    // TODO
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
    */
    assert(l->next->next->next);
    auto argl = l->next->next->data;
    auto body = l->next->next->next->data;
    auto params = map(argl, car);
    auto args = map(argl, cadr);
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
    auto call_fn = scm_concat_list2(scm_list1(func_args), args);
    auto new_expr = scm_list3(scm_sym_letrec(), scm_list1(letrec_arg), call_fn);
    if (debug_enabled) {
      printf("expand -> ");
      print_ast(new_expr);
      printf("\n");
    }
    return new_expr;
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
  auto argl = car(l->next->data);
  // auto param = car(argl);
  // auto arg = cadr(argl);
  auto rest_argl = cdr(l->next->data);
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
    // TODO:
  }
  else {
    assert(is_pair(func_args));
    auto argl = cast<SCM_List>(func_args);
    while (argl) {
      assert(argl->data);
      assert(is_pair(argl->data));
      auto arg = cast<SCM_List>(argl->data);
      assert(arg->next);

      auto new_param = make_list(scm_list2(arg->data, scm_bool_false()));
      params->next = new_param;
      params = params->next;

      auto new_arg = make_list(scm_list3(scm_sym_set(), arg->data, arg->next->data));
      args->next = new_arg;
      args = args->next;
      argl = argl->next;
    }
  }
  auto new_l = make_list(scm_sym_let());
  new_l->next = make_list(wrap(dummy_params.next));
  if (dummy_args.next) {
    new_l->next->next = dummy_args.next;
    args->next = l->next->next;
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