;; Test: re-export when two used modules export the same symbol
(set! %load-path (cons "." %load-path))
(load "rexport_a.scm")
(load "rexport_b.scm")
(load "rexport_consumer.scm")

(display "Testing re-export...") (newline)
(display (module-ref (resolve-module '(test rexport consumer)) 'shared-name)) (newline)
(display "REEXPORT-PASS") (newline)
