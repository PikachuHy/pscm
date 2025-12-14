;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test eq?, eqv?, and equal? with pairs and lists

;; Test eq? with pairs (should use pointer equality)
(define p1 (cons 1 2))
(define p2 (cons 1 2))
;; CHECK: #f
(eq? p1 p2)
;; CHECK-NEXT: #t
(eq? p1 p1)

;; Test eqv? with pairs (should use pointer equality, same as eq?)
;; CHECK: #f
(eqv? p1 p2)
;; CHECK-NEXT: #t
(eqv? p1 p1)

;; Test equal? with pairs (should use deep comparison)
;; CHECK: #t
(equal? p1 p2)
;; CHECK-NEXT: #t
(equal? p1 p1)

;; Test eq? with lists (should use pointer equality)
(define l1 (list 'a))
(define l2 (list 'a))
;; CHECK: #f
(eq? l1 l2)
;; CHECK-NEXT: #t
(eq? l1 l1)

;; Test eqv? with lists (should use pointer equality, same as eq?)
;; CHECK: #f
(eqv? l1 l2)
;; CHECK-NEXT: #t
(eqv? l1 l1)

;; Test equal? with lists (should use deep comparison)
;; CHECK: #t
(equal? l1 l2)
;; CHECK-NEXT: #t
(equal? l1 l1)

;; Test memq with lists (should use eq? - pointer equality)
;; CHECK: #f
(memq (list 'a) '(b (a) c))

;; Test member with lists (should use equal? - deep comparison)
;; CHECK: ((a) c)
(member (list 'a) '(b (a) c))

;; Test eqv? with dotted pairs (should use pointer equality)
(define d1 (cons 1 2))
(define d2 (cons 1 2))
;; CHECK: #f
(eqv? d1 d2)
;; CHECK-NEXT: #t
(eqv? d1 d1)

;; Test equal? with dotted pairs (should use deep comparison)
;; CHECK: #t
(equal? d1 d2)
;; CHECK-NEXT: #t
(equal? d1 d1)

;; Test equal? with nested dotted pairs
;; CHECK: #t
(equal? '(a . (b . (c . (d . (e . ()))))) '(a b c d e))

;; Test equal? with dotted pairs at end
;; CHECK: #t
(equal? '(a . (b . (c . d))) '(a b c . d))

;; Test eq? with same list object
(define x '(a))
;; CHECK: ((a) y)
(memq x (list 'x x 'y))

