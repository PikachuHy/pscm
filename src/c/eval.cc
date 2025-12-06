#include "pscm.h"

SCM_List _make_list_dummy() {
  SCM_List dummy;
  dummy.data = nullptr;
  dummy.next = nullptr;
  return dummy;
}

SCM *eval_with_list(SCM_Environment *env, SCM_List *l) {
  assert(l);
  // print_list(l);
  SCM *ret = nullptr;
  while (l) {
    ret = eval_with_env(env, l->data);
    l = l->next;
  }
  return ret;
}

SCM_List *eval_list_with_env(SCM_Environment *env, SCM_List *l) {
  SCM_List dummy = _make_list_dummy();
  SCM_List *it = &dummy;
  while (l) {
    auto val = eval_with_env(env, l->data);
    auto next = make_list(val);
    it->next = next;
    it = it->next;
    l = l->next;
  }
  return dummy.next;
}

SCM *eval_with_func_1(SCM_Function *func, SCM *arg1) {
  typedef SCM *(*func_1)(SCM *);
  auto f = (func_1)func->func_ptr;
  return f(arg1);
}

SCM *eval_with_func_2(SCM_Function *func, SCM *arg1, SCM *arg2) {
  typedef SCM *(*func_2)(SCM *, SCM *);
  auto f = (func_2)func->func_ptr;
  return f(arg1, arg2);
}

SCM *eval_with_func(SCM_Function *func, SCM_List *l) {
  if (debug_enabled) {
    SCM_DEBUG_EVAL("eval func ");
    printf("%s with ", func->name->data);
    print_list(l->next);
    printf("\n");
  }
  if (func->n_args == 1) {
    assert(l->next);
    auto arg1 = l->next->data;
    auto ret = eval_with_func_1(func, arg1);
    return ret;
  }
  if (func->n_args == 2) {
    assert(l);
    assert(l->next);
    auto arg1 = l->next->data;
    l = l->next;
    assert(l->next);
    auto arg2 = l->next->data;
    auto ret = eval_with_func_2(func, arg1, arg2);
    return ret;
  }
  if (func->n_args == -1 && func->generic) {
    auto ret = reduce(
        [func](SCM *lhs, SCM *rhs) {
          return eval_with_func_2(func, lhs, rhs);
        },
        func->generic, l->next);
    return ret;
  }
  else {
    printf("%s:%d not supported ", __FILE__, __LINE__);
    printf("function: %s", func->name->data);
    printf("\n");
    exit(1);
    return NULL;
  }
}

SCM *eval_with_env(SCM_Environment *env, SCM *ast) {
  assert(env);
  assert(ast);
entry:
  SCM_DEBUG_EVAL("eval ");
  if (debug_enabled) {
    print_ast(ast);
    printf("\n");
  }
  if (!is_pair(ast)) {
    if (is_sym(ast)) {
      SCM_Symbol *sym = cast<SCM_Symbol>(ast);
      auto val = scm_env_search(env, sym);
      if (!val) {
        SCM_ERROR_EVAL("symbol '%s' not found", sym->data);
        printf("\n");
        exit(1);
      }
      assert(val);
      return val;
    }
    return ast;
  }
  SCM_List *l = cast<SCM_List>(ast);
  assert(l->data);
  if (is_sym(l->data)) {
    SCM_Symbol *sym = cast<SCM_Symbol>(l->data);
    if (strcmp(sym->data, "define") == 0) {
      if (l->next && is_sym(l->next->data)) {
        SCM_Symbol *varname = cast<SCM_Symbol>(l->next->data);
        SCM_DEBUG_EVAL("define variable %s\n", varname->data);
        auto val = eval_with_env(env, l->next->next->data);
        assert(val);
        if (is_proc(val)) {
          auto proc = cast<SCM_Procedure>(val);
          if (proc->name == nullptr) {
            proc->name = varname;
            SCM_DEBUG_EVAL("define proc from lambda\n");
          }
        }
        scm_env_insert(env, varname, val);
        return scm_none();
      }
      else {
        SCM_List *proc_sig = cast<SCM_List>(l->next->data);
        SCM_DEBUG_EVAL("define a procedure");
        if (is_sym(proc_sig->data)) {
          SCM_Symbol *proc_name = cast<SCM_Symbol>(proc_sig->data);
          SCM_DEBUG_EVAL(" %s with params ", proc_name->data);
          if (debug_enabled) {
            printf("(");
            if (proc_sig->next) {
              print_ast(proc_sig->next->data);
            }
            printf(")\n");
          }
          // handle procedure body
          auto proc = make_proc(proc_name, proc_sig->next, l->next->next, env);
          SCM *ret = wrap(proc);
          scm_env_insert(env, proc_name, ret);
          return ret;
        }
        else {
          printf("%s:%d not supported ", __FILE__, __LINE__);
          print_ast(proc_sig->data);
          printf("\n");
          exit(1);
        }
      }
    }
    else if (strcmp(sym->data, "let") == 0) {
      ast = expand_let(ast);
      goto entry;
    }
    else if (strcmp(sym->data, "let*") == 0) {
      ast = expand_letstar(ast);
      goto entry;
    }
    else if (strcmp(sym->data, "letrec") == 0) {
      ast = expand_letrec(ast);
      goto entry;
    }
    else if (strcmp(sym->data, "call/cc") == 0 || strcmp(sym->data, "call-with-current-continuation") == 0) {
      assert(l->next);
      auto proc = eval_with_env(env, l->next->data);
      assert(is_proc(proc) || is_cont(proc) || is_func(proc));
      int first;
      auto cont = scm_make_continuation(&first);
      SCM_DEBUG_CONT("jump back: ");
      if (!first) {
        if (debug_enabled) {
          printf("cont is ");
          print_ast(cont);
          printf("\n");
        }
        // ast = cont;
        // goto entry;
        return cont;
      }
      else {
        auto new_l = scm_list2(proc, cont);
        if (debug_enabled) {
          print_ast(new_l);
          printf("\n");
        }
        ast = new_l;
        goto entry;
      }
#if 0
      auto cont = SCM_Continuation * cont;
      SCM_INIT_CONT(cont, cont_base);
      // {
      //   cont = new SCM_Continuation();
      //   long __stack_top__;
      //   cont->dst = &__stack_top__;
      //   cont->stack_len = (long)cont_base - (long)cont->dst;
      //   SCM_DEBUG_CONT("stack_len: %zu\n", cont->stack_len);
      //   cont->stack_data = (long *)malloc(sizeof(long) * cont->stack_len);
      //   memcpy((void *)cont->stack_data, (void *)cont->dst, sizeof(long) * cont->stack_len);
      // }
      // while (0)
      //   ;
      int ret = setjmp(cont->cont_jump_buffer);
      if (ret) {
        SCM_DEBUG_CONT("jump back\n");
        // print_ast(cont->arg);
        // exit(0);
        return cont->arg;
      }
      else {
        SCM_DEBUG_CONT("after setjmp\n");
        auto arg = l->next->data;
        auto cont_arg = new SCM();
        cont_arg->type = SCM::CONT;
        cont_arg->value = cont;

        auto f = eval_with_env(env, arg);
        assert(f);

        if (is_cont(cont_arg)) {
          SCM_DEBUG_EVAL("before is cont\n");
        }
        ast = scm_list2(f, cont_arg);
      }
#endif
    }
    else if (strcmp(sym->data, "lambda") == 0) {
      auto proc_sig = cast<SCM_List>(l->next->data);
      auto proc = make_proc(nullptr, proc_sig, l->next->next, env);
      auto ret = wrap(proc);

      if (debug_enabled) {
        SCM_DEBUG_EVAL("create proc ");
        print_ast(ret);
        printf(" from ");
        print_list(l);
        printf("\n");
      }
      return ret;
    }
    else if (strcmp(sym->data, "set!") == 0) {
      assert(is_sym(l->next->data));
      auto sym = cast<SCM_Symbol>(l->next->data);
      auto val = eval_with_env(env, l->next->next->data);
      assert(val);
      scm_env_insert(env, sym, val);
      if (debug_enabled) {
        SCM_DEBUG_EVAL("set! ");
        printf("%s to ", sym->data);
        print_ast(val);
        printf("\n");
      }
      return scm_nil();
    }
    else if (strcmp(sym->data, "quote") == 0) {
      if (l->next) {
        return l->next->data;
      }
      return scm_nil();
    }
    else if (strcmp(sym->data, "if") == 0) {
      assert(l->next);
      auto pred = eval_with_env(env, l->next->data);
      assert(is_bool(pred));
      if (is_true(pred)) {
        ast = l->next->next->data;
        goto entry;
      }
      else if (l->next->next->next) {
        ast = l->next->next->next->data;
        goto entry;
      }
      return scm_none();
    }
    else if (strcmp(sym->data, "cond") == 0) {
      /*
(cond
  ((> x 0) 'positive)
  ((< x 0) 'negative)
  (else 'zero))
      */
      assert(l->next);
      auto it = l->next;
      while (it) {
        assert(is_pair(it->data));
        auto clause = cast<SCM_List>(it->data);
        if (debug_enabled) {
          SCM_DEBUG_EVAL("eval cond clause ");
          print_list(clause);
          printf("\n");
        }
        if (is_sym_val(clause->data, "else")) {
          return eval_with_list(env, clause->next);
        }
        auto pred = eval_with_env(env, clause->data);
        if (is_bool(pred) && is_false(pred)) {
          it = it->next;
          continue;
        }
        if (!clause->next) {
          return scm_bool_true();
        }
        if (!is_sym_val(clause->next->data, "=>")) {
          return eval_with_list(env, clause->next);
        }
        auto val = scm_env_exist(env, cast<SCM_Symbol>(clause->next->data));
        if (val) {
          return eval_with_list(env, clause->next);
        }
        assert(clause->next->next);
        ast = scm_list2(clause->next->next->data, scm_list2(scm_sym_quote(), pred));
        goto entry;
      }
      return scm_none();
    }
    else if (strcmp(sym->data, "for-each") == 0) {
      // (for-each f arg)
      assert(l->next);
      auto f = eval_with_env(env, l->next->data);
      assert(is_proc(f));
      auto proc = cast<SCM_Procedure>(f);
      int arg_count = 0;
      auto l2 = proc->args;
      while (l2) {
        arg_count++;
        l2 = l2->next;
      }
      SCM_List dummy = _make_list_dummy();
      l2 = &dummy;
      l = l->next->next;

      SCM_List args_dummy = _make_list_dummy();
      auto args_iter = &args_dummy;

      for (int i = 0; i < arg_count; i++) {
        if (l) {
          auto item = make_list(eval_with_env(env, l->data));
          assert(is_pair(item->data));
          l2->next = item;
          l2 = item;

          l = l->next;

          auto arg = make_list();
          args_iter->next = arg;
          args_iter = arg;
        }
        else {
          SCM_ERROR_EVAL("args count not match, require %d, but got %d", arg_count, i + 1);
          exit(1);
        }
      }
      args_dummy.data = f;
      // do for-each
      assert(arg_count == 1);
      assert(is_pair(dummy.next->data));
      auto arg_l = cast<SCM_List>(dummy.next->data);
      while (arg_l) {
        args_dummy.next->data = arg_l->data;
        SCM t;
        t.type = SCM::LIST;
        t.value = &args_dummy;
        if (debug_enabled) {
          SCM_DEBUG_EVAL("for-each ")
          print_ast(f);
          printf(" ");
          print_ast(arg_l->data);
          printf("\n");
        }
        eval_with_env(env, &t);
        arg_l = arg_l->next;
      }
    }
    else if (strcmp(sym->data, "do") == 0) {
      /*
      (do ((<var1> <init1> <step1>)
          ...)
          (<test1> <expr> ...)
          <cmd> ...)
      */
      assert(l->next);
      assert(l->next->next);
      assert(l->next->next->next);
      auto var_init_l = cast<SCM_List>(l->next->data);
      auto test_clause = l->next->next->data;
      auto body_clause = l->next->next->next;

      if (debug_enabled) {
        SCM_DEBUG_EVAL("eval do\n");
        printf("var: ");
        print_list(var_init_l);
        printf("\n");
        printf("test: ");
        print_ast(test_clause);
        printf("\n");
        printf("cmd: ");
        print_list(body_clause);
        printf("\n");
      }
      auto do_env = make_env(env);

      auto var_init_it = var_init_l;
      SCM_List var_update_dummy = _make_list_dummy();

      auto var_update_it = &var_update_dummy;
      while (var_init_it) {
        // assert(is_pair(var_init_it->data));
        auto var_init_expr = cast<SCM_List>(var_init_it->data);
        auto var_name = cast<SCM_Symbol>(var_init_expr->data);
        auto var_init_val = eval_with_env(env, var_init_expr->next->data);
        auto var_update_step = var_init_expr->next->next->data;

        scm_env_insert(env, var_name, var_init_val);

        var_update_it->next = make_list(scm_list2(wrap(var_name), var_update_step));
        var_update_it = var_update_it->next;
        var_update_it->next = NULL;

        var_init_it = var_init_it->next;
      }

      auto ret = eval_with_env(do_env, car(test_clause));
      while (is_false(ret)) {
        eval_list_with_env(do_env, body_clause);

        var_update_it = &var_update_dummy;

        while (var_update_it->next) {
          var_update_it = var_update_it->next;
          auto var_update_expr = cast<SCM_List>(var_update_it->data);
          auto var_name = cast<SCM_Symbol>(var_update_expr->data);
          auto var_update_step = var_update_expr->next->data;

          if (debug_enabled) {
            SCM_DEBUG_EVAL("eval do step ... ");
            print_ast(var_update_step);
            printf("\n");
          }
          auto new_var_val = eval_with_env(do_env, var_update_step);
          if (debug_enabled) {
            SCM_DEBUG_EVAL("eval do step ... ");
            print_ast(var_update_step);
            printf(" --> ");
            print_ast(new_var_val);
            printf("\n");
          }
          scm_env_insert(do_env, var_name, new_var_val);
        }

        ret = eval_with_env(do_env, car(test_clause));
      }
      if (is_nil(cdr(test_clause))) {
        return scm_none();
      }
      return scm_none();
    }
    else {
      auto val = scm_env_search(env, sym);
      if (!val) {
        printf("%s:%d Symbol not found '%s'\n", __FILE__, __LINE__, sym->data);
        exit(1);
      }
      auto new_list = make_list(val);
      new_list->next = l->next;
      ast = wrap(new_list);
      goto entry;
    }
  }
  else if (is_cont(l->data)) {
    auto cont = cast<SCM_Continuation>(l->data);
    assert(cont);
    if (l->next) {
      assert(l->next);
      assert(l->next->data);
      auto cont_arg = eval_with_env(env, l->next->data);
      scm_dynthrow(l->data, cont_arg);
    }
    else {
      scm_dynthrow(l->data, scm_nil());
    }
  }
  else if (is_proc(l->data)) {
    auto proc = cast<SCM_Procedure>(l->data);
    auto proc_env = make_env(proc->env);
    auto args_l = proc->args;
    while (l->next && args_l) {
      assert(is_sym(args_l->data));
      auto arg_sym = cast<SCM_Symbol>(args_l->data);
      auto arg_val = eval_with_env(env, l->next->data);
      scm_env_insert(proc_env, arg_sym, arg_val, /*search_parent=*/false);
      if (debug_enabled) {
        SCM_DEBUG_EVAL("bind func arg ");
        printf("%s to ", arg_sym->data);
        print_ast(arg_val);
        printf("\n");
      }
      l = l->next;
      args_l = args_l->next;
    }
    if (l && args_l) {
      printf("args not match\n");
      printf("expect ");
      print_list(proc->args);
      printf("\n");
      printf("but got ");
      print_list(l->next);
      printf("\n");
      exit(1);
    }
    auto val = eval_with_list(proc_env, proc->body);
    return val;
  }
  else if (is_func(l->data)) {
    auto func = cast<SCM_Function>(l->data);
    auto func_argl = eval_list_with_env(env, l->next);
    if (debug_enabled) {
      SCM_DEBUG_EVAL(" ");
      printf("before eval args: ");
      print_list(l->next);
      printf("\n");
      printf("after eval args: ");
      print_list(func_argl);
      printf("\n");
    }
    l->next = func_argl;
    auto val = eval_with_func(func, l);
    return val;
  }
  else if (is_pair(l->data)) {
    auto f = eval_with_env(env, l->data);
    auto new_l = make_list(f);
    new_l->next = l->next;
    ast = wrap(new_l);
    goto entry;
  }
  else {
    printf("%s:%d not supported ", __FILE__, __LINE__);
    print_list(l);
    printf("\n");
    exit(1);
  }
  return scm_nil();
}
