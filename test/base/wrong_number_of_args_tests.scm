;; RUN: %pscm_cc --test %s | FileCheck %s
;; Test wrong-number-of-args error handling

;; Test 1: Catch wrong-number-of-args when calling car with no arguments
(catch 'wrong-number-of-args
       (lambda () (car))
       (lambda (type caller message opts extra)
         type))
;; CHECK: wrong-number-of-args

;; Test 2: Catch wrong-number-of-args when calling car with too many arguments
(catch 'wrong-number-of-args
       (lambda () (car 1 2))
       (lambda (type caller message opts extra)
         type))
;; CHECK-NEXT: wrong-number-of-args

;; Test 3: Catch wrong-number-of-args when calling cdr with no arguments
(catch 'wrong-number-of-args
       (lambda () (cdr))
       (lambda (type caller message opts extra)
         type))
;; CHECK-NEXT: wrong-number-of-args

;; Test 4: Check error message format
(catch 'wrong-number-of-args
       (lambda () (car))
       (lambda (type caller message opts extra)
         message))
;; CHECK-NEXT: "Wrong number of arguments to ~A"

;; Test 5: Check caller is #f
(catch 'wrong-number-of-args
       (lambda () (car))
       (lambda (type caller message opts extra)
         caller))
;; CHECK-NEXT: #f

;; Test 6: Check procedure is passed correctly
(catch 'wrong-number-of-args
       (lambda () (car))
       (lambda (type caller message opts extra)
         (if (procedure? opts)
             'procedure
             'not-procedure)))
;; CHECK-NEXT: procedure

;; Test 7: Check extra is #f
(catch 'wrong-number-of-args
       (lambda () (car))
       (lambda (type caller message opts extra)
         extra))
;; CHECK-NEXT: #f

;; Test 8: Test old-format? function (checks if message contains %)
(define old-format?
  (catch 'wrong-number-of-args
	 (lambda () (car))
	 (lambda (type caller message opts extra)
	   (let next ((l (string->list message)))
	     (cond ((null? l) #f)
		   ((char=? #\% (car l)) #t)
		   (else (next (cdr l))))))))
old-format?
;; CHECK-NEXT: #f

;; Test 9: Normal execution when no error occurs
(catch 'wrong-number-of-args
       (lambda () (car '(1 2 3)))
       (lambda (type caller message opts extra)
         'error))
;; CHECK-NEXT: 1

;; Test 10: Catch with #t tag (catches all errors including wrong-number-of-args)
(catch #t
       (lambda () (car))
       (lambda (type caller message opts extra)
         type))
;; CHECK-NEXT: wrong-number-of-args

;; Test 11: Test with cons (requires 2 arguments)
(catch 'wrong-number-of-args
       (lambda () (cons))
       (lambda (type caller message opts extra)
         type))
;; CHECK-NEXT: wrong-number-of-args

;; Test 12: Test with cons (too many arguments)
(catch 'wrong-number-of-args
       (lambda () (cons 1 2 3))
       (lambda (type caller message opts extra)
         type))
;; CHECK-NEXT: wrong-number-of-args

;; Test 13: Test with list-ref (requires 2 arguments)
(catch 'wrong-number-of-args
       (lambda () (list-ref))
       (lambda (type caller message opts extra)
         type))
;; CHECK-NEXT: wrong-number-of-args

;; Test 14: Test with list-ref (missing second argument)
(catch 'wrong-number-of-args
       (lambda () (list-ref '(1 2 3)))
       (lambda (type caller message opts extra)
         type))
;; CHECK-NEXT: wrong-number-of-args
