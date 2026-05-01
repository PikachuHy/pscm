;; RUN: %pscm_cc --test %s | FileCheck %s

(define saved #f)
(define result #f)
(call/cc (lambda (k) (set! saved k) (set! result 'first)))

;; CHECK: first
result

;; CHECK: ()
(gc)

;; CHECK: second
(saved 'second)
