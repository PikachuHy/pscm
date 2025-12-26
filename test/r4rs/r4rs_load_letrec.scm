(display "Start...")
(if (not (load "init.scm")) (exit 1))
(display "Load init.scm OK") (newline)

(define cur-section '())
(define errs '())

(define (SECTION . args)
  (display "SECTION") (write args) (newline)
  (set! cur-section args) #t)

(define (record-error e)
  (set! errs (cons (list cur-section e) errs)))

(define (test expect fun . args)
  (write (cons fun args))
  (display " ==> ")

  ((lambda (res)
     (write res)
     (newline)
     (cond ((not (equal? expect res))
            (record-error (list res expect (cons fun args)))
            (display " BUT EXPECTED ")
            (write expect)
            (newline)
            #f)
           (else #t)))

   (if (procedure? fun)
       (begin
         (write args)
         (newline)
         (apply fun args))
       (car args))))

(define (report-errs)
  (newline)
  (if (null? errs) (display "Passed all tests")
      (begin
        (display "errors were:")
        (newline)
        (display "(SECTION (got expected (call)))")
        (newline)
        (for-each (lambda (l) (write l) (newline)) errs)
	(exit 1)
	))
  (newline))
(SECTION 4 2 2)
(test #t 'letrec (letrec ((even?
                           (lambda (n) (if (zero? n) #t (odd? (- n 1)))))
                          (odd?
                           (lambda (n) (if (zero? n) #f (even? (- n 1))))))
                   (even? 88)))
(report-errs)
