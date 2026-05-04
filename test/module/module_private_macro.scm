;; Test: module-private variables are visible during macro expansion.
;; This verifies the environment chain module fix -- macro transformers
;; defined inside a module can reference that module's private (define) bindings.

(define-module (test macro-private) #:export (use-private))

;; Module-private variable -- NOT exported
(define secret-value "module-secret")

;; Module-private function -- NOT exported
(define (make-message x)
  (string-append "MSG:" x))

;; Macro whose transformer references module-private bindings
(define-macro (use-private)
  `(string-append ,make-message ,secret-value))

;; Now use the macro (within the same module)
(display (use-private)) (newline)
(display "PASS") (newline)
