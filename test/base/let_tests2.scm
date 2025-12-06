;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; RUN: %pscm_cc --test %s | FileCheck %s
;; RUN: %pscm_main --test %s | FileCheck %s

(define (factorial n)
  (let loop ((k n) (acc 1))
    (if (<= k 1)
        acc
        (loop (- k 1) (* acc k)))))

;; CHECK: 2
(factorial 2)
;; CHECK: 6
(factorial 3)
;; CHECK: 120
(factorial 5)
;; CHECK: 3628800
(factorial 10)
;; CHECK: 2432902008176640000
(factorial 20)
;; TODO: support bignum
