;; RUN: %pscm_cc --test %s | FileCheck %s
;; Test scm_call_* and scm_apply_0 functions (indirectly through normal procedure calls)
;; Note: These are C API functions, so we test their behavior indirectly

;; Test 1: Procedure with 0 arguments (tests scm_call_0 behavior)
(define proc0 (lambda () 42))
;; CHECK: 42
(proc0)

;; Test 2: Procedure with 1 argument (tests scm_call_1 behavior)
(define proc1 (lambda (x) x))
;; CHECK-NEXT: 10
(proc1 10)

;; Test 3: Procedure with 2 arguments (tests scm_call_2 behavior)
(define proc2 (lambda (x y) (+ x y)))
;; CHECK-NEXT: 30
(proc2 10 20)

;; Test 4: Procedure with 3 arguments (tests scm_call_3 behavior)
(define proc3 (lambda (x y z) (+ x (+ y z))))
;; CHECK-NEXT: 60
(proc3 10 20 30)

;; Test 5: Apply with empty list (tests scm_apply_0 behavior with nil)
(define proc-apply (lambda args (if (null? args) 'empty (car args))))
;; CHECK-NEXT: empty
(apply proc-apply '())

;; Test 6: Apply with list of arguments (tests scm_apply_0 behavior)
;; CHECK-NEXT: 100
(apply + '(10 20 30 40))

;; Test 7: Apply with single element list
;; CHECK-NEXT: 42
(apply proc1 '(42))

;; Test 8: Apply with multiple elements
;; CHECK-NEXT: 6
(apply * '(1 2 3))

;; Test 9: Nested apply (tests scm_apply_0 in nested context)
(define add-all (lambda args (apply + args)))
;; CHECK-NEXT: 15
(apply add-all '(1 2 3 4 5))

;; Test 10: Apply with procedure that takes rest parameter
(define sum (lambda args (apply + args)))
;; CHECK-NEXT: 10
(sum 1 2 3 4)

;; Test 11: Apply with procedure that takes rest parameter and apply
;; CHECK-NEXT: 20
(apply sum '(5 5 5 5))

;; Test 12: Call with builtin function (tests scm_call_* with functions)
;; CHECK-NEXT: 3
(+ 1 2)

;; Test 13: Call with builtin function, 0 args (tests scm_call_0 with functions)
;; This tests that functions can be called with 0 args
;; (Actually, most builtin functions require args, so we test with a procedure)
(define zero-arg-proc (lambda () 'zero))
;; CHECK-NEXT: zero
(zero-arg-proc)

