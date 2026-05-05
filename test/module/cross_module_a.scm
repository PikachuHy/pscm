(define-module (cross module a)
  #:export (value-a helper-func make-double))

(define value-a 42)
(define (helper-func x) (+ x 1))
(define (make-double x) (* x 2))
