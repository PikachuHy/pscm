;; define-module option parsing test
;; Verify #:use-module and #:export inside define-module body

(display "Test 1: define-module with #:use-module\n")
(define-module (options-source))
(define-public src-val 42)
(define-public (src-func x) (+ x 1))

(define-module (options-consumer)
  #:use-module (options-source)
  #:export consumer-val)

(display "PASS: define-module with options\n")

(display "\nTest 2: #:use-module imports bindings\n")
(if (= src-val 42)
    (display "PASS: #:use-module imports variable\n")
    (begin
      (display "FAIL: #:use-module did not import variable\n")
      (exit 1)))

(if (= (src-func 5) 6)
    (display "PASS: #:use-module imports function\n")
    (begin
      (display "FAIL: #:use-module did not import function\n")
      (exit 1)))

(display "\nTest 3: #:pure flag\n")
(define-module (pure-test) #:pure)
(display "PASS: define-module with #:pure\n")

(display "\nAll define-module options tests passed!\n")
