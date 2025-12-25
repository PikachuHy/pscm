;; Module lookup tests

;; Test 1: module-bound?
(display "Test 1: module-bound?\n")
(define-module (lookup-test))
(define test-var 42)

;; Create symbol for lookup
(define test-var-sym (string->symbol "test-var"))
(if (module-bound? (current-module) test-var-sym)
    (display "PASS: module-bound? finds defined variable\n")
    (begin
      (display "FAIL: module-bound? does not find defined variable\n")
      (exit 1)))

(if (not (module-bound? (current-module) 'nonexistent))
    (display "PASS: module-bound? returns #f for nonexistent variable\n")
    (begin
      (display "FAIL: module-bound? does not return #f for nonexistent variable\n")
      (exit 1)))

;; Test 2: module-ref
(display "\nTest 2: module-ref\n")
(define ref-value (module-ref (current-module) test-var-sym))
(if (= ref-value 42)
    (display "PASS: module-ref retrieves variable value\n")
    (begin
      (display "FAIL: module-ref does not retrieve variable value\n")
      (exit 1)))

;; Test 3: module-ref with default
(display "\nTest 3: module-ref with default\n")
(define nonexistent-sym (string->symbol "nonexistent"))
(define default-value (module-ref (current-module) nonexistent-sym 'default))
(if (eq? default-value 'default)
    (display "PASS: module-ref returns default for nonexistent variable\n")
    (begin
      (display "FAIL: module-ref does not return default for nonexistent variable\n")
      (exit 1)))

(display "\nAll lookup tests passed!\n")

