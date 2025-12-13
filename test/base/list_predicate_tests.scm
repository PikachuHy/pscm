;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test the list? predicate function
;; list? returns #t if the argument is a proper list (ends with nil), #f otherwise

;; Test list? with nil (empty list is a proper list)
;; CHECK: #t
(list? '())

;; Test list? with single element list
;; CHECK: #t
(list? '(a))

;; Test list? with multiple element list
;; CHECK: #t
(list? '(a b c))

;; Test list? with nested list
;; CHECK: #t
(list? '(a (b c) d))

;; Test list? with dotted pair (not a proper list)
;; CHECK: #f
(list? '(a . b))

;; Test list? with improper list (dotted pair at end)
;; CHECK: #f
(list? '(a b . c))

;; Test list? with number (not a list)
;; CHECK: #f
(list? 3)

;; Test list? with symbol (not a list)
;; CHECK: #f
(list? 'a)

;; Test list? with string (not a list)
;; CHECK: #f
(list? "hello")

;; Test list? with vector (not a list)
;; CHECK: #f
(list? #(1 2 3))

;; Test list? with boolean (not a list)
;; CHECK: #f
(list? #t)

;; Test list? with #f (not a list)
;; CHECK: #f
(list? #f)

;; Test list? with list created by list function
;; CHECK: #t
(list? (list 'a 'b 'c))

;; Test list? with list after set-cdr!
;; This tests that list? correctly identifies proper lists even after mutation
(define x (list 'a 'b 'c))
;; CHECK: #t
(list? x)
(set-cdr! x 4)
;; CHECK: #f
(list? x)

;; Test list? with empty list created by list function
;; CHECK: #t
(list? (list))

;; Test list? with list containing various types
;; CHECK: #t
(list? '(1 "hello" #t 'symbol))

