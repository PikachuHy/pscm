;; RUN: %pscm_cc --test %s | FileCheck %s

;; These tests focus on macros that work with rest arguments and
;; make sure dotted parameter lists and rest args are preserved
;; correctly through macro expansion.

;; 1. Macro that wraps its arguments into a call using quasiquote
;;    and unquote-splicing.
(define-macro (call-with-list f . xs)
  `(,f ,@xs))

;; CHECK: (a b c)
(call-with-list list 'a 'b 'c)

;; 3. Macro whose transformer itself uses a dotted parameter list.
;;    This is similar to the R5RS test: the transformer takes a
;;    fixed prefix and a rest list, then emits code that returns
;;    (rest ...).
(define-macro (suffix-args x y . z)
  ;; Expand to a lambda that ignores x/y and returns the rest args.
  `(lambda (,x ,y . tail) tail))

;; CHECK: (5 6)
((suffix-args a b c d) 3 4 5 6)


