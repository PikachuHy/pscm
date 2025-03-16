;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; RUN: %pscm_main --test %s | FileCheck %s

(define (a b) 1)
;; CHECK: 1
(eval '(a 1))
