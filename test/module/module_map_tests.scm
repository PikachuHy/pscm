;; Module map tests

;; Test 1: Basic module-map functionality
(display "Test 1: Basic module-map\n")
(define-module (test-map))
(define var1 10)
(define var2 20)
(define var3 30)

;; Map over module to get all symbols
(define all-symbols (module-map (lambda (sym var) sym) (current-module)))
(if (pair? all-symbols)
    (display "PASS: module-map returns list\n")
    (begin
      (display "FAIL: module-map does not return list\n")
      (exit 1)))

;; Test 2: module-map with value extraction
(display "\nTest 2: module-map with value extraction\n")
(define all-values (module-map (lambda (sym var) var) (current-module)))
(if (pair? all-values)
    (display "PASS: module-map returns values\n")
    (begin
      (display "FAIL: module-map does not return values\n")
      (exit 1)))

;; Test 3: module-map with custom procedure
(display "\nTest 3: module-map with custom procedure\n")
(define symbol-count (module-map (lambda (sym var) 1) (current-module)))
;; Count the results
(define count 0)
(define count-list symbol-count)
(if (pair? count-list)
    (begin
      (set! count (+ count 1))
      (set! count-list (cdr count-list)))
    (begin))
(if (> count 0)
    (display "PASS: module-map works with custom procedure\n")
    (begin
      (display "FAIL: module-map does not work with custom procedure\n")
      (exit 1)))

;; Test 4: module-map on module with uses
(display "\nTest 4: module-map on module with uses\n")
(define-module (test-uses))
;; Define some variables in this module
(define uses-var1 100)
(define uses-var2 200)
(use-modules (test-map))
;; module-map should only include symbols from current module's obarray, not from uses
(define uses-symbols (module-map (lambda (sym var) sym) (current-module)))
;; Should include symbols from current module (uses-var1, uses-var2)
(if (pair? uses-symbols)
    (display "PASS: module-map works on module with uses\n")
    (begin
      (display "FAIL: module-map does not work on module with uses\n")
      (exit 1)))

(display "\nAll module-map tests passed!\n")

