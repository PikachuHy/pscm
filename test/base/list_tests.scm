;; RUN: %pscm_cc --test %s | FileCheck %s

;; CHECK: ()
(append)
;; CHECK: ()
(list)

(define a '(a b c d))
;; CHECK: (a b c d)
a

(define b (list-head a 2))
;; CHECK: (a b)
b
(set-cdr! b 2)
;; CHECK: (a . 2)
b
;; CHECK: (a b c d)
a

(define a '(a b c d))
(define b (list-tail a 2))
;; CHECK: (c d)
b
(set-cdr! b 2)
;; CHECK: (c . 2)
b

(define a '(a b c d))
(define b (list-tail a 3))
;; CHECK: (d)
b

;; CHECK: (d)
(last-pair '(a b c d))
;; CHECK: ()
(last-pair '())

;; Test memq function
;; CHECK: (a b c)
(memq 'a '(a b c))

;; CHECK: (b c)
(memq 'b '(a b c))

;; CHECK: (c)
(memq 'c '(a b c))

;; CHECK: #f
(memq 'd '(a b c))

;; CHECK: #f
(memq 'a '())

;; Test memq with numbers (should use eq? comparison)
;; CHECK: (2 3)
(memq 2 '(1 2 3))

;; CHECK: #f
(memq 4 '(1 2 3))

;; Test memq with symbols
;; CHECK: (foo bar)
(memq 'foo '(baz foo bar))

;; CHECK: #f
(memq 'qux '(baz foo bar))

;; Test memq with lists
;; Note: In this implementation, eq? uses deep equality for lists
;; So (memq '(a) '(b (a) c)) will match because eq? compares list contents
;; CHECK: ((a) c)
(memq '(a) '(b (a) c))

;; Test memq with the same list object (pointer equality)
(define x '(a))
;; CHECK: ((a) y)
(memq x (list 'x x 'y))

;; Test memq with characters
;; CHECK: (#\b #\c)
(memq #\b '(#\a #\b #\c))

;; CHECK: #f
(memq #\d '(#\a #\b #\c))