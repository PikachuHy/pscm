;; Custom binder test
;; Verify set-module-binder! and binder invocation

(display "Test 1: create module and set binder\n")
(define-module (binder-test))
(define my-binder
  (lambda (module sym definep)
    (display "binder called for: ") (display sym) (newline)
    (if (eq? sym 'dynamic-val)
        (make-variable 999)
        #f)))
(set-module-binder! (current-module) my-binder)
(display "PASS: set-module-binder!\n")

(display "\nTest 2: binder provides dynamic-val\n")
(if (= dynamic-val 999)
    (display "PASS: binder provides dynamic-val\n")
    (begin
      (display "FAIL: binder did not provide dynamic-val\n")
      (exit 1)))

(display "\nTest 3: binder returns #f for unknown symbol\n")
(define unknown-accessible #t)
(set! unknown-accessible
  (catch #t
    (lambda () undefined-sym #t)
    (lambda (key . args) #f)))
(if (not unknown-accessible)
    (display "PASS: unknown symbol not provided by binder\n")
    (begin
      (display "FAIL: binder returned something for unknown symbol\n")
      (exit 1)))

(display "\nAll binder tests passed!\n")
