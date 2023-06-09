(display (current-module)) (newline)
(define-module (test))
(display (current-module)) (newline)
(display "???") (newline)
(define *tests-run* 0)
(define *tests-passed* 0)


(define-macro (test . args)
  ;(display (length args)) (newline)
  ;(display args) (newline)
  (let ((expr (car args))
	(expect (cadr args)))
    ;(display expr) (newline)
    ;(display expect) (newline)
    (if (= (length args) 3)
	(begin
	  (set! expr (cadr args))
	  (set! expect (caddr args))))
    `(begin
     (set! *tests-run* (+ *tests-run* 1))
     (let ((str (call-with-output-string
		 (lambda (out)
		   (write *tests-run*)
		   (display ". ")
		   (display ',expr out))))
	   (res ,expr))
       (display str)
       (write-char #\space)
       (display (make-string (max 0 (- 72 (string-length str))) #\.))
       ;(flush-output)
       (cond
	((equal? res ,expect)
	 (set! *tests-passed* (+ *tests-passed* 1))
	 (display " [PASS]\n"))
	(else
	 (display " [FAIL]\n")
	 (display "    expected ") (write ,expect)
	 (display " but got ") (write res) (newline)))))))

(define-macro (test-assert expr)
  `(test #t ,expr))

(define (test-begin . name)
  #f)

(define (test-end)
(display (current-module)) (newline)
(display "*tests-passed*: ") (display *tests-passed*) (newline)
  (write *tests-passed*)
  (display " out of ")
  (write *tests-run*)
  (display " passed (")
  (write (* (/ *tests-passed* *tests-run*) 100.0))
  (display "%)")
  (if (< *tests-passed* *tests-run*) (exit 1))
  (newline))

(export test-begin test *tests-run* *tests-passed* test-end)
(display "Done!!!") (newline)