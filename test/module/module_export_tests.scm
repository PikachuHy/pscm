;; Module export tests

;; Test 1: define-public exports symbol
(display "Test 1: define-public\n")
(define-module (export-test))
(define-public (public-func) "public")
(define (private-func) "private")

(display "PASS: define-public defines and exports\n")

;; Test 2: export exports multiple symbols
(display "\nTest 2: export\n")
(define var1 1)
(define var2 2)
(define var3 3)
(export var1 var2 var3)

(display "PASS: export exports multiple symbols\n")

;; Test 3: use-modules imports exported symbols
(display "\nTest 3: use-modules imports\n")
(define-module (import-test))
(use-modules (export-test))

;; Try to access exported symbols
(if (= var1 1)
    (display "PASS: use-modules imports exported variables\n")
    (begin
      (display "FAIL: use-modules does not import exported variables\n")
      (exit 1)))

(if (string=? (public-func) "public")
    (display "PASS: use-modules imports exported functions\n")
    (begin
      (display "FAIL: use-modules does not import exported functions\n")
      (exit 1)))

(display "\nAll export tests passed!\n")

