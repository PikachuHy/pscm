;; Basic module system tests

;; Test 1: current-module returns a module
(display "Test 1: current-module\n")
(define mod1 (current-module))
(display mod1) (newline)
(if (module? mod1)
    (display "PASS: current-module returns a module\n")
    (begin
      (display "FAIL: current-module does not return a module\n")
      (display "mod1 type check failed\n")
      (exit 1)))

;; Test 2: define-module creates a new module
(display "\nTest 2: define-module\n")
(define-module (test-module))
(define mod2 (current-module))
(display mod2) (newline)
(if (module? mod2)
    (display "PASS: define-module creates a module\n")
    (begin
      (display "FAIL: define-module does not create a module\n")
      (exit 1)))

;; Test 3: define in module
(display "\nTest 3: define in module\n")
(define test-var 42)
(display test-var) (newline)
(if (= test-var 42)
    (display "PASS: define works in module\n")
    (begin
      (display "FAIL: define does not work in module\n")
      (exit 1)))

;; Test 4: export and use-modules
(display "\nTest 4: export and use-modules\n")
(define-module (module-a))
(define-public (hello) "hello from module-a")
(define-public (add x y) (+ x y))

(define-module (module-b))
(use-modules (module-a))
(display (hello)) (newline)
(if (string=? (hello) "hello from module-a")
    (display "PASS: use-modules imports exported functions\n")
    (begin
      (display "FAIL: use-modules does not import exported functions\n")
      (exit 1)))

(if (= (add 2 3) 5)
    (display "PASS: use-modules imports exported functions with arguments\n")
    (begin
      (display "FAIL: use-modules does not import exported functions with arguments\n")
      (exit 1)))

;; Test 5: %load-path
(display "\nTest 5: %load-path\n")
(display "%load-path: ") (display %load-path) (newline)
(set! %load-path (append %load-path '(".")))
(display "%load-path after append: ") (display %load-path) (newline)
(if (pair? %load-path)
    (display "PASS: %load-path is accessible and mutable\n")
    (begin
      (display "FAIL: %load-path is not accessible or mutable\n")
      (exit 1)))

(display "\nAll basic module tests passed!\n")

