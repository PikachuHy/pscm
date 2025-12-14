;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test eq?, eqv?, and equal? functions

;; Test eq? with numbers (should compare by value)
;; CHECK: #t
(eq? 1 1)
;; CHECK: #f
(eq? 1 2)
;; CHECK: #t
(eq? 1.0 1.0)
;; CHECK: #t
(eq? 1 1.0)

;; Test eq? with symbols (should compare by content)
;; CHECK: #t
(eq? 'a 'a)
;; CHECK: #f
(eq? 'a 'b)

;; Test eq? with lists (uses pointer equality in R4RS)
(define a (list 'a))
(define b (list 'a))
(define c a)
;; CHECK: #t
(eq? a a)
;; CHECK: #t
(eq? a c)
;; CHECK: #f
(eq? a b)  ; Pointer equality - different objects

;; Test eq? with strings (should compare by content)
;; CHECK: #t
(eq? "hello" "hello")
;; CHECK: #f
(eq? "hello" "world")

;; Test eqv? with numbers (should compare by value, same as eq?)
;; CHECK: #t
(eqv? 1 1)
;; CHECK: #f
(eqv? 1 2)
;; CHECK: #t
(eqv? 1.0 1.0)
;; CHECK: #t
(eqv? 1 1.0)

;; Test eqv? with lists (uses pointer equality in R4RS, same as eq?)
;; CHECK: #t
(eqv? a a)
;; CHECK: #t
(eqv? a c)
;; CHECK: #f
(eqv? a b)  ; Pointer equality - different objects

;; Test equal? with numbers (should compare by value)
;; CHECK: #t
(equal? 1 1)
;; CHECK: #f
(equal? 1 2)
;; CHECK: #t
(equal? 1.0 1.0)
;; CHECK: #t
(equal? 1 1.0)

;; Test equal? with lists (should do deep comparison)
;; CHECK: #t
(equal? a a)
;; CHECK: #t
(equal? a c)
;; CHECK: #t
(equal? a b)
;; CHECK: #t
(equal? '(1 2 3) '(1 2 3))
;; CHECK: #f
(equal? '(1 2 3) '(1 2 4))
;; CHECK: #t
(equal? '((1 2) (3 4)) '((1 2) (3 4)))
;; CHECK: #f
(equal? '((1 2) (3 4)) '((1 2) (3 5)))

;; Test equal? with nested structures
;; CHECK: #t
(equal? '(a (b c) d) '(a (b c) d))
;; CHECK: #f
(equal? '(a (b c) d) '(a (b d) d))

;; Test equal? with vectors
;; CHECK: #t
(equal? '#(1 2 3) '#(1 2 3))
;; CHECK: #f
(equal? '#(1 2 3) '#(1 2 4))

;; Test equal? with strings
;; CHECK: #t
(equal? "hello" "hello")
;; CHECK: #f
(equal? "hello" "world")

;; Test with characters
;; CHECK: #t
(eq? #\a #\a)
;; CHECK: #f
(eq? #\a #\b)
;; CHECK: #t
(eqv? #\a #\a)
;; CHECK: #t
(equal? #\a #\a)

;; Test with booleans
;; CHECK: #t
(eq? #t #t)
;; CHECK: #t
(eq? #f #f)
;; CHECK: #f
(eq? #t #f)
;; CHECK: #t
(eqv? #t #t)
;; CHECK: #t
(equal? #t #t)

;; Test with empty lists
;; CHECK: #t
(eq? '() '())
;; CHECK: #t
(eqv? '() '())
;; CHECK: #t
(equal? '() '())

;; Test memq behavior (uses eq? for comparison - pointer equality)
;; CHECK: #f
(memq (list 'a) '(b (a) c))  ; Pointer equality - different objects

;; Test member behavior (uses equal? for comparison)
;; CHECK: ((a) c)
(member (list 'a) '(b (a) c))  ; Deep comparison

;; Test with same list object
(define x '(a))
;; CHECK: ((a) y)
(memq x (list 'x x 'y))

