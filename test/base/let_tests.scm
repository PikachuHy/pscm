;; RUN: %pscm_cc --test %s | FileCheck %s

(define x '(1 3 5 7 9))

(define sum 0)

(do ((x x (cdr x)))
    ((null? x))
    (set! sum (+ sum (car x))))

;; CHECK: 25
sum

;; CHECK: 25
(let ((x '(1 3 5 7 9))
      (sum 0))
  (do ((x x (cdr x)))
    ((null? x))
    (set! sum (+ sum (car x))))
  sum)

;; CHECK: 9
(letrec () (define x 9) x)

;; CHECK: 6
(let ((x 2) (y 3))
  (* x y))

;; CHECK: 35
(let ((x 2) (y 3))
  (let ((x 7)
        (z (+ x y)))
    (* z x)))

;; CHECK: 70
(let ((x 2) (y 3))
    (let* ((x 7)
          (z (+ x y)))
      (* z x)))

;; CHECK: 3
(let* ((a 1)
      (b 2))
 (+ a b))
