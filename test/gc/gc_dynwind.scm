;; RUN: %pscm_cc --test %s | FileCheck %s

(define result '())
(call/cc
  (lambda (k)
    (dynamic-wind
      (lambda () (set! result (cons 'in result)))
      (lambda () (gc) (set! result (cons 'body result)))
      (lambda () (set! result (cons 'out result))))))

;; CHECK: 3
(length result)
