;; Test: call/cc across module boundaries
(set! %load-path (cons "." %load-path))
(load "cont_a.scm")
(load "cont_b.scm")
(display (capture)) (newline)
(display "CONT-PASS") (newline)
