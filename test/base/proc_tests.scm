;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; RUN: %pscm_main --test %s | FileCheck %s
;; RUN: %pscm_cc --test %s | FileCheck %s

;; CHECK: #f
(noop)

(define (a b) 2)

;; CHECK: a
(procedure-name a)

(define b a)

;; CHECK: a
(procedure-name b)

(define c (lambda () '2))

;; CHECK: c
(procedure-name c)

(define ((c b) a) '2)

;; CHECK: #<procedure #f (a)>
(c 2)

;; CHECK: 2
((c 2) 3)

(define (factorial n)
  (define (iter product counter)
    (if (> counter n)
        product
        (iter (* counter product)
              (+ counter 1))))
  (iter 1 1))

;; CHECK: 720
(factorial 6)

(define ((define-property* which) opt decl)
  `(,which (list ,@opt)))

;; CHECK: #<procedure #f (opt decl)>
(define-property* 'a)

;; CHECK: (a (list c d))
((define-property* 'a) '(c d) 'e)

;; CHECK: 6
((lambda () 6))

;; CHECK: 6
((lambda () (* 2 3)))

;; CHECK: 12
((lambda () (* 2 3) (* 3 4)))


(define (f return)
  (return 2)
  3)
;; CHECK: 3
(f (lambda (x) x))

;; Test lambda with single symbol parameter (rest parameter)
;; CHECK: (3 4 5 6)
((lambda x x) 3 4 5 6)

;; CHECK: ()
((lambda x x))

;; CHECK: (a b c)
((lambda args args) 'a 'b 'c)

;; Test lambda with normal parameter list (should still work)
;; CHECK: 7
((lambda (x y) (+ x y)) 3 4)

;; Test lambda with rest parameter (dotted pair syntax)
;; CHECK: (5 6)
((lambda (x y . z) z) 3 4 5 6)

;; CHECK: ()
((lambda (x y . z) z) 3 4)