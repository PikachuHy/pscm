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

;; Test make-vector function
;; CHECK: #(none none none none none)
(make-vector 5)

;; CHECK: #(hi hi hi hi hi)
(make-vector 5 'hi)

;; CHECK: #()
(make-vector 0)

;; CHECK: #()
(make-vector 0 'a)

;; CHECK: #(0 0 0)
(make-vector 3 0)

;; Test vector-length function
;; CHECK: 5
(vector-length (make-vector 5))

;; CHECK: 0
(vector-length '#())

;; CHECK: 3
(vector-length '#(a b c))

;; CHECK: 1
(vector-length '#(x))

;; Test vector-ref function
;; CHECK: a
(vector-ref '#(a b c) 0)

;; CHECK: b
(vector-ref '#(a b c) 1)

;; CHECK: c
(vector-ref '#(a b c) 2)

;; CHECK: 1
(vector-ref '#(1 2 3) 0)

;; CHECK: hi
(vector-ref (make-vector 2 'hi) 0)

;; CHECK: hi
(vector-ref (make-vector 2 'hi) 1)

;; Test vector-set! function
(define vec1 (make-vector 5))
(vector-set! vec1 0 0)
(vector-set! vec1 1 1)
(vector-set! vec1 2 2)
(vector-set! vec1 3 3)
(vector-set! vec1 4 4)
;; CHECK: #(0 1 2 3 4)
vec1

(define vec2 '#(a b c))
(vector-set! vec2 1 'x)
;; CHECK: #(a x c)
vec2

(define vec3 (vector 0 '(2 2 2 2) "Anna"))
(vector-set! vec3 1 '("Sue" "Sue"))
;; CHECK: #(0 ("Sue" "Sue") "Anna")
vec3

;; Test vector function
;; CHECK: #()
(vector)

;; CHECK: #(a)
(vector 'a)

;; CHECK: #(a b c)
(vector 'a 'b 'c)

;; CHECK: #(1 2 3 4 5)
(vector 1 2 3 4 5)

;; CHECK: #(a 1 "test" #t)
(vector 'a 1 "test" #t)

;; Test vector->list function
;; CHECK: (a b c)
(vector->list '#(a b c))

;; CHECK: ()
(vector->list '#())

;; CHECK: (dah dah didah)
(vector->list '#(dah dah didah))

;; CHECK: (1 2 3)
(vector->list '#(1 2 3))

;; Test list->vector function
;; CHECK: #(dididit dah)
(list->vector '(dididit dah))

;; CHECK: #()
(list->vector '())

;; CHECK: #(a b c)
(list->vector '(a b c))

;; CHECK: #(1 2 3)
(list->vector '(1 2 3))

;; Test round-trip conversion
;; CHECK: #(a b c)
(list->vector (vector->list '#(a b c)))

;; CHECK: (a b c)
(vector->list (list->vector '(a b c)))

;; Test vector operations in do loop
;; Note: This test is commented out due to a bug in do loop implementation
;; TODO: Fix do loop to properly handle vector-set! in body
;; (do ((vec (make-vector 5))
;;      (i 0 (+ i 1)))
;;     ((= i 5) vec)
;;   (vector-set! vec i i))

;; Test vector-ref with computed indices
(define vec4 '#(10 20 30 40 50))
;; CHECK: 10
(vector-ref vec4 0)
;; CHECK: 30
(vector-ref vec4 2)
;; CHECK: 50
(vector-ref vec4 4)

;; Test vector-set! with computed indices
(define vec5 (make-vector 3))
(vector-set! vec5 0 'first)
(vector-set! vec5 1 'second)
(vector-set! vec5 2 'third)
;; CHECK: #(first second third)
vec5

;; Test vector with mixed types
(define vec6 (vector 'symbol 42 "string" #t #f))
;; CHECK: #(symbol 42 "string" #t #f)
vec6
;; CHECK: symbol
(vector-ref vec6 0)
;; CHECK: 42
(vector-ref vec6 1)
;; CHECK: "string"
(vector-ref vec6 2)
;; CHECK: #t
(vector-ref vec6 3)
;; CHECK: #f
(vector-ref vec6 4)
