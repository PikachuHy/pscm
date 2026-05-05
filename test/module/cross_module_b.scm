(define-module (cross module b)
  #:use-module (cross module a)
  #:export (value-b b-func))

(define value-b (* value-a 2))
(define (b-func x) (helper-func (make-double x)))
