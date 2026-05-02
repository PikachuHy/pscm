;; Autoload lazy loading test
;; Verify that #:autoload only loads module on first access

;; Ensure current directory is in load path so autoload-target.scm is found
(set! %load-path (cons "." %load-path))

(display "Test 1: define module with autoload\n")
(define-module (autoload-consumer)
  #:autoload (autoload-target) lazy-func)

(display "PASS: define-module with #:autoload\n")

(display "\nTest 2: lazy-func should work (triggers autoload)\n")
(if (= (lazy-func 7) 21)
    (display "PASS: autoload lazy-func works\n")
    (begin
      (display "FAIL: autoload lazy-func does not work\n")
      (exit 1)))

(display "\nTest 3: lazy-func available after autoload\n")
(if (= (lazy-func 2) 6)
    (display "PASS: autoload symbol persists\n")
    (begin
      (display "FAIL: autoload symbol did not persist\n")
      (exit 1)))

(display "\nAll autoload tests passed!\n")
