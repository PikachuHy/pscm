;; RUN: %pscm_cc --test %s | FileCheck %s

;; CHECK: "Hello World"
"Hello World"

'(#t => 'ok)

#t

;; Test string=? function
;; CHECK: #t
(string=? "hello" "hello")
;; CHECK: #f
(string=? "hello" "world")
;; CHECK: #t
(string=? "a" "a")
;; CHECK: #f
(string=? "abc" "ab")
;; CHECK: #f
(string=? "ab" "abc")
;; CHECK: #t
(string=? "" "")
;; CHECK: #f
(string=? "hello" "Hello")

;; Test string function (create string from characters)
;; CHECK: ""
(string)
;; CHECK: "abc"
(string #\a #\b #\c)
;; CHECK: "Hello"
(string #\H #\e #\l #\l #\o)
;; CHECK: "123"
(string #\1 #\2 #\3)
;; CHECK: "a b"
(string #\a #\space #\b)