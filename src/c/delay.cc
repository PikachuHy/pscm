#include "pscm.h"
#include "eval.h"

extern SCM_Environment g_env;

// (delay <expr>): create a promise that will evaluate <expr> when forced
SCM *eval_delay(SCM_Environment *env, SCM_List *l) {
  // l is the whole list: (delay <expr>)
  // l->data is the symbol 'delay', so the expression is in l->next
  if (!l->next || !l->next->data) {
    eval_error("delay: expected one expression");
  }

  SCM *expr = l->next->data;

  // Create a zero-argument procedure (thunk) whose body is (expr)
  SCM_List *args = nullptr;          // empty parameter list
  SCM_List *body = make_list(expr);  // body contains a single expression

  // Name can be nullptr for anonymous thunk
  SCM_Procedure *proc = make_proc(nullptr, args, body, env);
  SCM *thunk = wrap(proc);

  // Create the promise object
  SCM_Promise *promise = new SCM_Promise();
  promise->thunk = thunk;
  promise->value = nullptr;
  promise->is_forced = false;

  SCM *ret = new SCM();
  ret->type = SCM::PROMISE;
  ret->value = promise;
  ret->source_loc = nullptr;

  return ret;
}

// force: force a promise, computing and caching its value
//
// Semantics follow R5RS `make-promise` to correctly handle *re-entrant* force:
// - If a promise is forced while it is already in the middle of being forced,
//   the inner force "wins" and its value is cached.
// - The outer force then returns the cached value, discarding its own
//   intermediate result.
//
// This ensures examples like the following evaluate to 3 (not 4):
//   (letrec ((p (delay (if c
//                       3
//                       (begin
//                         (set! c #t)
//                         (+ (force p) 1)))))
//            (c #f))
//     (force p))
//
// which matches R4RS/R5RS test suites and reference implementations.
SCM *scm_c_force(SCM *promise_scm) {
  if (!is_promise(promise_scm)) {
    type_error(promise_scm, "promise");
  }

  SCM_Promise *promise = cast<SCM_Promise>(promise_scm);

  // Fast path: already forced
  if (promise->is_forced) {
    return promise->value;
  }

  if (!promise->thunk) {
    eval_error("force: invalid promise (missing thunk)");
  }

  SCM *thunk = promise->thunk;

  // Evaluate the thunk *without* marking the promise as forced yet.
  // This allows re-entrant `force` calls during evaluation to set the
  // final value first, just like the R5RS `make-promise` reference
  // implementation.
  SCM *result = nullptr;
  if (is_proc(thunk)) {
    SCM_Procedure *proc = cast<SCM_Procedure>(thunk);
    // Call procedure with no arguments in the environment where it was created
    SCM_List *no_args = nullptr;
    SCM_Environment *proc_env = proc->env ? proc->env : &g_env;
    result = apply_procedure(proc_env, proc, no_args);
  } else if (is_func(thunk)) {
    SCM_Function *func = cast<SCM_Function>(thunk);
    // Build a call list: (thunk)
    SCM_List call;
    call.data = thunk;
    call.next = nullptr;
    call.is_dotted = false;
    result = eval_with_func(func, &call);
  } else {
    eval_error("force: invalid promise thunk");
  }

  // Re-entrant `force` may have completed and set the final value while
  // we were evaluating `thunk`. In that case, do not overwrite it; just
  // return the cached value.
  if (promise->is_forced) {
    return promise->value;
  }

  // Normal, non-re-entrant case: cache the result now.
  promise->value = result;
  promise->is_forced = true;
  return result;
}

// Initialize delay/force primitives
void init_delay() {
  scm_define_function("force", 1, 0, 0, scm_c_force);
}


