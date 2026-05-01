;; RUN: %pscm_cc --test %s | FileCheck %s

;; CHECK: ()
;; CHECK: 1
;; CHECK: 2
(define x (cons 1 2))
(gc)
(car x)
(cdr x)
