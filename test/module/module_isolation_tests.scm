;; Module isolation tests

;; Test 1: Variables in different modules are isolated
(display "Test 1: Module isolation\n")
(define-module (module-x))
(define x-var 100)

(define-module (module-y))
(define x-var 200)

(if (= x-var 200)
    (display "PASS: Modules have isolated namespaces\n")
    (begin
      (display "FAIL: Modules do not have isolated namespaces\n")
      (exit 1)))

;; Test 2: use-modules imports from other modules
(display "\nTest 2: use-modules imports\n")
(define-module (module-z))
(use-modules (module-x))
;; After use-modules, x-var from module-x should be accessible
(if (= x-var 100)
    (display "PASS: use-modules imports variables from other modules\n")
    (begin
      (display "FAIL: use-modules does not import variables from other modules\n")
      (exit 1)))

;; Test 3: Local definition shadows imported
(display "\nTest 3: Local definition shadows imported\n")
(define-module (module-w))
(use-modules (module-x))
(define x-var 300)  ;; Local definition should shadow imported
(if (= x-var 300)
    (display "PASS: Local definition shadows imported variable\n")
    (begin
      (display "FAIL: Local definition does not shadow imported variable\n")
      (exit 1)))

(display "\nAll isolation tests passed!\n")

