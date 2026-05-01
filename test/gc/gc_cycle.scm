;; RUN: %pscm_cc --test %s | FileCheck %s

;; CHECK: ()
;; CHECK: 1
(define l (list 1 2 3))
(set-cdr! (cddr l) l)
(gc)
(car l)
