(define-module (autoload-target))
(define-public lazy-func
  (lambda (x) (* x 3)))
(display "autoload-target module loaded\n")
