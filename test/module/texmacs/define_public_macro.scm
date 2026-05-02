;; Test: define-public-macro (TeXmacs uses this to export macros)
;; define-public-macro is basically define-public + define-macro combined
;; In Guile 1.8 it defines a macro AND exports it
(define-module (test dpm) #:export (my-when))
(define-macro (my-when test . body)
  `(if ,test (begin ,@body)))
;; Now test it works within the same module
(my-when #t (display "works") (newline))
(display "DPM-PASS") (newline)
