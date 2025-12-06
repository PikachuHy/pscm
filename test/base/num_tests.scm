;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; RUN: %pscm_cc --test %s | FileCheck %s --check-prefix=LONGJMP
;; RUN: %pscm_cc --test %s | FileCheck %s
;; TODO: %pscm_main --test %s | FileCheck %s

;; LONGJMP-NOT: not supported
;; LONGJMP: #<primitive-generic +>
+
;; CHECK: 0
(+)
;; CHECK: 666
(+ 666)
;; CHECK: 3
(+ 1 2)
;; CHECK: 6
(+ 1 2 3)
;; CHECK: 10
(+ 1 2 3 4)
;; CHECK: 6
(* 2 3)

(define a 3)
(define b 6)
;; CHECK: 18
(* a b)

(define (add a b) (+ a b))
;; CHECK: 4
(add 1 3)
