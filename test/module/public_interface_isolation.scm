;; Public interface isolation test
;; Verify that use-modules only imports exported symbols

(display "Test 1: define-module with export\n")
(define-module (isolation-mod))
(define pub-x 10)
(define priv-y 20)
(export pub-x)

(display "PASS: define-module and export\n")

(display "\nTest 2: use-modules imports exported symbol\n")
(define-module (isolation-consumer))
(use-modules (isolation-mod))
(if (= pub-x 10)
    (display "PASS: use-modules imports exported symbol\n")
    (begin
      (display "FAIL: use-modules does not import exported symbol\n")
      (exit 1)))

(display "\nTest 3: use-modules does not import non-exported symbol\n")
;; priv-y was not exported, so it should be unbound
(define priv-accessible #t)
(set! priv-accessible
  (catch #t
    (lambda () priv-y #t)
    (lambda (key . args) #f)))
(if (not priv-accessible)
    (display "PASS: non-exported symbol is not accessible\n")
    (begin
      (display "FAIL: non-exported symbol is accessible when it shouldn't be\n")
      (exit 1)))

(display "\nAll public interface isolation tests passed!\n")
