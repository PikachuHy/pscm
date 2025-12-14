;; RUN: %pscm_cc --test %s | FileCheck %s

(define (a b) 1)
;; CHECK: 1
(eval '(a 1))
