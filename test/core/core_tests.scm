;; RUN: %pscm_cc --test %s | FileCheck %s

;; CHECK: 5
(+ 2 3)

;; CHECK: 9
(+ 2 3 4)

;; CHECK: -2
(- 2)

;; CHECK: -1
(- 2 3)

(define (sum a b c)
  (+ a b c))
;; CHECK: 6
(sum 1 2 3)

;; CHECK: 100
(cond ((> 3 2) 100)
      ((< 3 2) 200))

;; CHECK: 34
;; 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
(define (fib n)
      (cond ((< n 2) 0)
            ((< n 4) 1)
            (else (+ (fib (- n 1)) (fib (- n 2))))))
    (fib 10)

(define (abs n)
    (cond ((< n 0) (- n))
          (else n)))
;; CHECK: 5
(abs -5)
;; CHECK: (4 5 6)
(map abs '(4 -5 6))

(define (map-fn list) (map abs list))
;; CHECK: (4 5 6)
(map-fn '(4 -5 6))

;; CHECK: 4
(car '(4 -5 6))
