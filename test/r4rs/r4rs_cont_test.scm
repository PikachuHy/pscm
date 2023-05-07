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

(SECTION 1 1)
(test #t call-with-current-continuation procedure?)
(test -3 call-with-current-continuation
      (lambda (exit)
        (for-each (lambda (x)
                    (if (negative? x)
                        (exit x)))
                  '(54 0 37 -3 245 19))
        #t))
(define list-length
  (lambda (obj)
    (call-with-current-continuation
     (lambda (return)
       (letrec ((r (lambda (obj) (cond ((null? obj) 0)
                                       ((pair? obj) (+ (r (cdr obj)) 1))
                                       (else (return #f))))))
         (r obj))))))

(test 4 list-length '(1 2 3 4))
(test #f list-length '(a b . c))

(report-errs)


;;; This tests full conformance of call-with-current-continuation.  It
;;; is a separate test because some schemes do not support call/cc
;;; other than escape procedures. I am indebted to
;;; raja@copper.ucs.indiana.edu (Raja Sooriamurthi) for fixing this
;;; code. The function leaf-eq? compares the leaves of 2 arbitrary
;;; trees constructed of conses.
(define (next-leaf-generator obj eot)
  (letrec ((return #f)
           (cont (lambda (x)
                   (recur obj)
                   (set! cont (lambda (x) (return eot)))
                   (cont #f)))
           (recur (lambda (obj)
                    (if (pair? obj)
                        (for-each recur obj)
                        (call-with-current-continuation
                         (lambda (c)
                           (set! cont c)
                           (return obj)))))))
    (lambda ()
      (call-with-current-continuation
       (lambda (ret)
         (set! return ret) (cont #f))))))

(define (leaf-eq? x y)
  (let* ((eot (list 'eot))
         (xf (next-leaf-generator x eot))
         (yf (next-leaf-generator y eot)))
    (letrec ((loop (lambda (x y)
                     (cond ((not (eq? x y)) #f)
                           ((eq? eot x) #t)
                           (else (loop (xf) (yf)))))))
      (loop (xf) (yf)))))

(define (test-cont)
  (newline)
  (display ";testing continuations; ")
  (newline)
  (SECTION 6 9)
  (test #t leaf-eq? '(a (b (c))) '((a) b c))
  (test #f leaf-eq? '(a (b (c))) '((a) b c d))
  (report-errs))

(test-cont)