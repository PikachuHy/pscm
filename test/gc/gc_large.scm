;; RUN: %pscm_cc --test %s | FileCheck %s

;; CHECK: ()
;; CHECK: 0
(define v (make-vector 1000 0))
(gc)
(vector-ref v 999)
