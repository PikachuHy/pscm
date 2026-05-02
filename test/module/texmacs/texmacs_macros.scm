;; Test: texmacs-module macro with all options TeXmacs uses
(set! %load-path (cons "." %load-path))
(load "boot.scm")
(load "m.scm")

;; Test :use option
(texmacs-module (test use)
  (:use (m)))

;; Test :inherit option
(texmacs-module (test inherit)
  (:inherit (m)))

(display (module-ref (resolve-module '(test use)) 'my-identity)) (newline)
(display (module-ref (resolve-module '(test inherit)) 'my-identity)) (newline)
(display "TEXMACSMACROS-PASS") (newline)
