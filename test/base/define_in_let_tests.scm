;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test that define creates local bindings inside let/let*/letrec
;; and does not affect outer environment variables

(define x 34)

;; Test define inside let with binding
;; CHECK: 5
(let ((x 3)) (define x 5) x)

;; CHECK: 34
x

;; Test define inside empty let
;; CHECK: 6
(let () (define x 6) x)

;; CHECK: 34
x

;; Test let with outer variable reference
;; CHECK: 34
(let ((x x)) x)

;; Test define inside let* with binding
;; CHECK: 7
(let* ((x 3)) (define x 7) x)

;; CHECK: 34
x

;; Test define inside empty let*
;; CHECK: 8
(let* () (define x 8) x)

;; CHECK: 34
x

;; Test define inside letrec with binding
;; CHECK: 9
(letrec () (define x 9) x)

;; CHECK: 34
x

;; Test define inside letrec with existing binding
;; CHECK: 10
(letrec ((x 3)) (define x 10) x)

;; CHECK: 34
x

;; Test nested let with define
;; CHECK: 11
(let ((x 2) (y 3))
  (let ((x 7))
    (define x 11)
    x))

;; CHECK: 34
x

;; Test define with same name as let binding
;; CHECK: 12
(let ((x 1))
  (define x 12)
  x)

;; CHECK: 34
x

;; Test multiple defines in let
;; CHECK: 13
(let ()
  (define a 13)
  (define b 14)
  a)

;; CHECK: 14
(let ()
  (define a 13)
  (define b 14)
  b)

;; Test define in letrec with mutual recursion
;; CHECK: #t
(letrec ((even?
          (lambda (n) (if (zero? n) #t (odd? (- n 1)))))
         (odd?
          (lambda (n) (if (zero? n) #f (even? (- n 1))))))
  (define test-value #t)
  (and (even? 88) test-value))

;; CHECK: 34
x

;; Test that define in let does not shadow outer define
(define y 100)

;; CHECK: 200
(let ()
  (define y 200)
  y)

;; CHECK: 100
y

;; Test define in let* with sequential defines
;; CHECK: 300
(let* ()
  (define a 300)
  (define b (+ a 1))
  a)

;; CHECK: 301
(let* ()
  (define a 300)
  (define b (+ a 1))
  b)

