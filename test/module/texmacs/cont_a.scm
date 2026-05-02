(define-module (test cont a) #:export (capture))
(define saved #f)
(define (capture)
  (call/cc (lambda (k) (set! saved k) 'captured)))
