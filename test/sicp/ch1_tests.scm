;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; RUN: %pscm_main --test %s | FileCheck %s

;; CHECK: 486
486

;; CHECK: 486
(+ 137 349)

;; CHECK: 666
(- 1000 334)

;; CHECK: 495
(* 5 99)

;; CHECK: 2
(/ 10 5)

;; CHECK: 12.7
(+ 2.7 10)

;; CHECK: 75
(+ 21 35 12 7)

;; CHECK: 1200
(* 25 4 12)

;; CHECK: 19
(+ (* 3 5) (- 10 6))

;; CHECK: 57
(+ (* 3 (+ (* 2 4) (+ 3 5))) (+ (- 10 7) 6))

(define size 2)
;; CHECK: 2
size
;; CHECK: 10
(* 5 size)

(define pi 3.14159)
(define radius 10)
;; CHECK: 314.159
(* pi (* radius radius))
(define circumference (* 2 pi radius))
;; CHECK: 62.8318
circumference

(define (square x) (* x x))
;; CHECK: #<procedure square (x)>
square
;; CHECK: 441
(square 21)
;; CHECK: 49
(square (+ 2 5))
;; CHECK: 81
(square (square 3))
(define (sum-of-squares x y)
  (+ (square x) (square y)))
;; CHECK: #<procedure sum-of-squares (x y)>
sum-of-squares
;; CHECK: 25
(sum-of-squares 3 4)

(define (f a)
  (sum-of-squares (+ a 1) (* a 2)))
;; CHECK: 136
(f 5)


(define (abs x)
  (cond ((> x 0) x)
        ((= x 0) 0)
        ((< x 0) (- x))))
;; CHECK: 1
(abs -1)
;; CHECK: 0
(abs 0)
;; CHECK: 1
(abs 1)

(define (abs x)
  (cond ((< x 0) (- x))
        (else x)))
;; CHECK: 1
(abs -1)
;; CHECK: 0
(abs 0)
;; CHECK: 1
(abs 1)

(define (abs x)
  (if (< x 0)
      (- x)
      x))
;; CHECK: 1
(abs -1)
;; CHECK: 0
(abs 0)
;; CHECK: 1
(abs 1)

(define x 1)
;; CHECK: #f
(and (> x 5) (< x 10))

(define x 5)
;; CHECK: #f
(and (> x 5) (< x 10))

(define x 7)
;; CHECK: #t
(and (> x 5) (< x 10))

(define x 10)
;; CHECK: #f
(and (> x 5) (< x 10))

(define x 11)
;; CHECK: #f
(and (> x 5) (< x 10))

(define (>= x y) (or (> x y) (= x y)))
;; CHECK: #t
(>= 2 2)
;; CHECK: #t
(>= 3 2)
;; CHECK: #f
(>= 1 2)

(define (>= x y) (not (< x y)))
;; CHECK: #t
(>= 2 2)
;; CHECK: #t
(>= 3 2)
;; CHECK: #f
(>= 1 2)


;; CHECK: 10
10
;; CHECK: 12
(+ 5 3 4)
;; CHECK: 8
(- 9 1)
;; CHECK: 4
(- 6 2)
;; CHECK: 6
(+ (* 2 4) (- 4 6))

(define a 3)
(define b (+ a 1))
;; CHECK: 19
(+ a b (* a b))
;; CHECK: #f
(= a b)
;; CHECK: 4
(if (and (> b a) (< b (* a b)))
   b
   a)
;; CHECK: 16
(cond ((= a 4) 6)
   ((= b 4) (+ 6 7 a))
   (else 25))

;; CHECK: 6
(+ 2 (if (> b a) b a))

;; CHECK: 16
(* (cond ((> a b) a)
         ((< a b) b)
         (else -1))
   (+ a 1))

(define (factorial n)
  (if (= n 1)
      1
      (* n (factorial (- n 1)))))
;; CHECK: 720
(factorial 6)

(define (factorial n)
  (fact-iter 1 1 n))
(define (fact-iter product counter max-count)
  (if (> counter max-count)
      product
      (fact-iter (* counter product)
                 (+ counter 1)
                 max-count)))
;; CHECK: 720
(factorial 6)

(define (factorial n)
  (define (iter product counter)
    (if (> counter n)
        product
        (iter (* counter product)
              (+ counter 1))))
  (iter 1 1))
;; CHECK: 720
(factorial 6)

(define (fib n)
  (cond ((= n 0) 0)
        ((= n 1) 1)
        (else (+ (fib (- n 1))
                 (fib (- n 2))))))
;; CHECK: 5
(fib 5)
;; CHECK: 8
(fib 6)
;; CHECK: 13
(fib 7)
;; CHECK: 21
(fib 8)

(define (fib n)
  (fib-iter 1 0 n))
(define (fib-iter a b count)
  (if (= count 0)
      b
      (fib-iter (+ a b) a (- count 1))))
;; CHECK: 5
(fib 5)
;; CHECK: 8
(fib 6)
;; CHECK: 13
(fib 7)
;; CHECK: 21
(fib 8)


