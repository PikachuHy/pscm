;; RUN: %pscm_cc --test %s | FileCheck %s

(define cc #f)
;; CHECK: #f
cc
;; CHECK: 14
(+ (call/cc (lambda (return) (set! cc return) (* 2 3)))(+ 1 7))
;; CHECK: #<continuation@
cc
;; CHECK: 18
(cc 10)
;; CHECK: 14
(cc (* 2 3))
