;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; TODO: %pscm_main --test %s | FileCheck %s

;; CHECK: 5
(call-with-values (lambda () (values 4 5))
         (lambda (a b) b))
;; CHECK: -1
(call-with-values * -)

(define (values . things)
  (call/cc
    (lambda (cont) (apply cont things))))
;; CHECK: 5
(call-with-values (lambda () (values 4 5))
    (lambda (a b) b))

;; CHECK: -1
(call-with-values * -)