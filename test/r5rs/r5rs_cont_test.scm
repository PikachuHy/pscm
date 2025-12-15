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
  (write *tests-passed*)
  (display " out of ")
  (write *tests-run*)
  (display " passed (")
  (write (* (/ *tests-passed* *tests-run*) 100.0))
  (display "%)")
  (if (< *tests-passed* *tests-run*) (exit 1))
  (newline))

(test #t (call-with-current-continuation procedure?))

(test 7 (call-with-current-continuation (lambda (k) (+ 2 5))))

(test 3 (call-with-current-continuation (lambda (k) (+ 2 5 (k 3)))))


(test '(a b c)
    (let* ((path '())
           (add (lambda (s) (set! path (cons s path)))))
      (dynamic-wind (lambda () (add 'a)) (lambda () (add 'b)) (lambda () (add 'c)))
      (reverse path)))

(test '(connect talk1 disconnect connect talk2 disconnect)
    (let ((path '())
          (c #f))
      (let ((add (lambda (s)
                   (set! path (cons s path)))))
        (dynamic-wind
            (lambda () (add 'connect))
            (lambda ()
              (add (call-with-current-continuation
                    (lambda (c0)
                      (set! c c0)
                      'talk1))))
            (lambda () (add 'disconnect)))
        (if (< (length path) 4)
            (c 'talk2)
            (reverse path)))))
