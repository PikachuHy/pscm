;; Test: module-use! then define should shadow the import
(set! %load-path (cons "." %load-path))
(load "usebind_target.scm")
(load "usebind_consumer.scm")
(display "USEBIND-PASS") (newline)
