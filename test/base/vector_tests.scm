;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test parsing of vectors

;; CHECK: #()
'#()

;; CHECK: #(a b c)
'#(a b c)

;; CHECK: #(1 2 3)
'#(1 2 3)

;; CHECK: #(#t #f)
'#(#t #f)

;; CHECK: #(a b c d e)
'#(a b c d e)

;; Test vectors in lists
;; CHECK: (#() #(a b c))
(list '#() '#(a b c))

;; CHECK: (#(1) #(2) #(3))
(list '#(1) '#(2) '#(3))

;; Test nested vectors
;; CHECK: #(#(a) #(b) #(c))
'#(#(a) #(b) #(c))

;; Test vectors with mixed types
;; CHECK: #(a 1 #t "test")
'#(a 1 #t "test")

;; Test vectors with symbols containing dots
;; CHECK: #(... !.. #:+.)
'#(... !.. :+.)

;; Test vectors in quoted expressions
(define empty-vec '#())
;; CHECK: #()
empty-vec

(define test-vec '#(a b c))
;; CHECK: #(a b c)
test-vec

;; Test vectors with empty vector
;; CHECK: #(#())
'#(#())

;; Test vectors with lists
;; CHECK: #((a b) (c d))
'#((a b) (c d))

;; Test vectors with dotted pairs
;; CHECK: #((a . b) (c . d))
'#((a . b) (c . d))

;; Test vectors with nested structures
;; CHECK: #((a b c) #(1 2 3) (x . y))
'#((a b c) #(1 2 3) (x . y))

;; Test vectors in complex expressions
;; CHECK: (#() #(a) #(a b) #(a b c))
(list '#() '#(a) '#(a b) '#(a b c))

;; Test vectors with special characters
;; CHECK: #(+ - * /)
'#(+ - * /)

;; Test vectors with numbers
;; CHECK: #(0 1 2 3 4 5)
'#(0 1 2 3 4 5)

;; Test vectors with negative numbers
;; CHECK: #(-1 -2 -3)
'#(-1 -2 -3)

;; Test vectors with strings
;; CHECK: #("hello" "world")
'#("hello" "world")

;; Test vectors with empty string
;; CHECK: #("" "test")
'#("" "test")

;; Test vectors with characters (if supported)
;; CHECK: #(#\a #\b #\c)
'#(#\a #\b #\c)

;; Test vectors with mixed content
;; CHECK: #(a 1 "test" #t #f)
'#(a 1 "test" #t #f)

;; Test vectors with nested vectors and lists
;; CHECK: #(#(a b) (c d) #(e f) (g h))
'#(#(a b) (c d) #(e f) (g h))
