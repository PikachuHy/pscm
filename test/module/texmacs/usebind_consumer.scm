(define-module (test usebind consumer))
(module-use! (resolve-module '(test usebind consumer)) (resolve-interface '(test usebind target)))
;; Shadow via local define
(define foo "local-foo")
(display foo) (newline)
