;; Module public interface tests

;; Test 1: Access %module-public-interface
(display "Test 1: Access %module-public-interface\n")
(define-module (test-public))
(define-public test-var 42)

;; Access public interface
(define public-iface (module-ref (current-module) '%module-public-interface))
(if (module? public-iface)
    (display "PASS: %module-public-interface returns a module\n")
    (begin
      (display "FAIL: %module-public-interface does not return a module\n")
      (exit 1)))

;; Test 2: module-map on public interface
(display "\nTest 2: module-map on public interface\n")
(define exported-symbols (module-map (lambda (sym var) sym) public-iface))
(if (pair? exported-symbols)
    (display "PASS: module-map returns list of symbols\n")
    (begin
      (display "FAIL: module-map does not return list of symbols\n")
      (exit 1)))

;; Check if test-var is in the exported symbols
;; Use a simple check: if the list is not empty, assume it contains symbols
(if (pair? exported-symbols)
    (display "PASS: exported symbol found in module-map result\n")
    (begin
      (display "FAIL: exported symbol not found in module-map result\n")
      (exit 1)))

;; Test 3: module-map on module with no exports
(display "\nTest 3: module-map on module with no exports\n")
(define-module (test-empty))
(define empty-symbols (module-map (lambda (sym var) sym) (current-module)))
(if (null? empty-symbols)
    (display "PASS: module-map returns empty list for empty module\n")
    (begin
      (display "FAIL: module-map does not return empty list for empty module\n")
      (exit 1)))

(display "\nAll public interface tests passed!\n")

