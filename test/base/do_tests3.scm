;; RUN: %pscm_cc --test %s | FileCheck %s

;; CHECK: #(0 1 2 3 4)
(do ((vec (make-vector 5))
     (i 0 (+ i 1)))
  ((= i 5) vec)
  (vector-set! vec i i))

;; CHECK: 25
(let ((x '(1 3 5 7 9)))
  (do ((x x (cdr x))
       (sum 0 (+ sum (car x))))
    ((null? x) sum)))
