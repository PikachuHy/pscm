;; RUN: %pscm_cc --test %s | FileCheck %s --check-prefix=LONGJMP
;; RUN: %pscm_cc --test %s | FileCheck %s

;; LONGJMP-NOT: not supported
;; LONGJMP: #<primitive-generic +>
+
;; CHECK: 0
(+)
;; CHECK: 666
(+ 666)
;; CHECK: 3
(+ 1 2)
;; CHECK: 6
(+ 1 2 3)
;; CHECK: 10
(+ 1 2 3 4)
;; CHECK: 6
(* 2 3)

(define a 3)
(define b 6)
;; CHECK: 18
(* a b)

(define (add a b) (+ a b))
;; CHECK: 4
(add 1 3)

;; Test zero? function
;; CHECK: #t
(zero? 0)

;; CHECK: #f
(zero? 1)

;; CHECK: #f
(zero? -1)

;; CHECK: #t
(zero? 0.0)

;; CHECK: #f
(zero? 0.5)

;; CHECK: #f
(zero? -0.5)

;; CHECK: #t
(zero? -0)

;; Test zero? with negative zero (should be true)
;; CHECK: #t
(zero? (- 0))

;; Test zero? in conditional expressions
;; CHECK: #t
(if (zero? 0) #t #f)

;; CHECK: #f
(if (zero? 1) #t #f)

;; Test positive? function
;; CHECK: #t
(positive? 5)
;; CHECK: #f
(positive? -5)
;; CHECK: #f
(positive? 0)
;; CHECK: #t
(positive? 3.5)
;; CHECK: #f
(positive? -3.5)
;; CHECK: #f
(positive? 0.0)

;; Test odd? function
;; CHECK: #t
(odd? 3)
;; CHECK: #f
(odd? 2)
;; CHECK: #t
(odd? -3)
;; CHECK: #f
(odd? -4)
;; CHECK: #f
(odd? 0)
;; CHECK: #t
(odd? 1)
;; CHECK: #f
(odd? 100)

;; Test even? function
;; CHECK: #f
(even? 3)
;; CHECK: #t
(even? 2)
;; CHECK: #f
(even? -3)
;; CHECK: #t
(even? -4)
;; CHECK: #t
(even? 0)
;; CHECK: #f
(even? 1)
;; CHECK: #t
(even? 100)

;; Test number type predicates
;; CHECK: #t
(complex? 3)
;; CHECK: #t
(complex? 3.5)
;; CHECK: #f
(complex? "hello")
;; CHECK: #t
(real? 3)
;; CHECK: #t
(real? 3.5)
;; CHECK: #f
(real? "hello")
;; CHECK: #t
(rational? 3)
;; CHECK: #f
(rational? 3.5)
;; CHECK: #f
(rational? "hello")
;; CHECK: #t
(integer? 3)
;; CHECK: #f
(integer? 3.5)
;; CHECK: #f
(integer? "hello")
;; CHECK: #t
(exact? 3)
;; CHECK: #f
(exact? 3.5)
;; CHECK: #f
(exact? "hello")
;; CHECK: #f
(inexact? 3)
;; CHECK: #t
(inexact? 3.5)
;; CHECK: #f
(inexact? "hello")
