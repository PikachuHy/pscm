;; Test: dynamic-wind with module-local state
(define-module (test wind m) #:export (test-wind))
(define wind-log '())
(define (test-wind)
  (dynamic-wind
    (lambda () (set! wind-log (cons 'before wind-log)))
    (lambda () (set! wind-log (cons 'during wind-log)) 'thunk-result)
    (lambda () (set! wind-log (cons 'after wind-log))))
  wind-log)

(display (test-wind)) (newline)
(display "DYNWIND-PASS") (newline)
