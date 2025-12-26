;; Test re-export functionality
;; re-export allows re-exporting symbols from other modules

;; Test 1: Basic re-export
(define-module (test-re-export-1))
(define var1 10)
(export var1)

(define-module (test-re-export-user-1))
(use-modules (test-re-export-1))
(display "var1 from module: ") (display var1) (newline)
(re-export var1)
(display "var1 re-exported") (newline)

;; Test 2: Re-export multiple symbols
(define-module (test-re-export-2))
(define func1 (lambda (x) (* x 2)))
(define func2 (lambda (x) (+ x 1)))
(define var2 20)
(export func1 func2 var2)

(define-module (test-re-export-user-2))
(use-modules (test-re-export-2))
(display "func1(5): ") (display (func1 5)) (newline)
(display "func2(5): ") (display (func2 5)) (newline)
(display "var2: ") (display var2) (newline)
(re-export func1 func2 var2)
(display "All symbols re-exported") (newline)

;; Test 3: Re-export from multiple modules
(define-module (test-re-export-3a))
(define a-var 100)
(export a-var)

(define-module (test-re-export-3b))
(define b-var 200)
(export b-var)

(define-module (test-re-export-user-3))
(use-modules (test-re-export-3a) (test-re-export-3b))
(display "a-var: ") (display a-var) (newline)
(display "b-var: ") (display b-var) (newline)
(re-export a-var b-var)
(display "Both symbols re-exported") (newline)

;; Test 4: Re-export procedure
(define-module (test-re-export-4))
(define (add-one x) (+ x 1))
(export add-one)

(define-module (test-re-export-user-4))
(use-modules (test-re-export-4))
(display "add-one(10): ") (display (add-one 10)) (newline)
(re-export add-one)
(display "add-one re-exported") (newline)

;; Test 5: Re-export after use-modules
(define-module (test-re-export-5))
(define exported-var 500)
(export exported-var)

(define-module (test-re-export-user-5))
(use-modules (test-re-export-5))
(re-export exported-var)
(display "exported-var re-exported: ") (display exported-var) (newline)

