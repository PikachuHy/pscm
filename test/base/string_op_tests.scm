;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test string operation functions

;; Test substring
;; CHECK: ""
(substring "ab" 0 0)
;; CHECK-NEXT: ""
(substring "ab" 1 1)
;; CHECK-NEXT: ""
(substring "ab" 2 2)
;; CHECK-NEXT: "a"
(substring "ab" 0 1)
;; CHECK-NEXT: "b"
(substring "ab" 1 2)
;; CHECK-NEXT: "ab"
(substring "ab" 0 2)
;; CHECK-NEXT: "abc"
(substring "abcdef" 0 3)
;; CHECK-NEXT: "def"
(substring "abcdef" 3 6)
;; CHECK-NEXT: "cde"
(substring "abcdef" 2 5)
;; CHECK-NEXT: "Hello"
(substring "Hello, World" 0 5)
;; CHECK-NEXT: "World"
(substring "Hello, World" 7 12)

;; Test string-append
;; CHECK: ""
(string-append)
;; CHECK-NEXT: "foobar"
(string-append "foo" "bar")
;; CHECK-NEXT: "foo"
(string-append "foo")
;; CHECK-NEXT: "foo"
(string-append "foo" "")
;; CHECK-NEXT: "foo"
(string-append "" "foo")
;; CHECK-NEXT: "abc"
(string-append "a" "b" "c")
;; CHECK-NEXT: "Hello, World!"
(string-append "Hello" ", " "World" "!")
;; CHECK-NEXT: "123456"
(string-append "1" "2" "3" "4" "5" "6")
;; CHECK-NEXT: ""
(string-append "" "" "")

;; Test string->list
;; CHECK: (#\P #\space #\l)
(string->list "P l")
;; CHECK-NEXT: ()
(string->list "")
;; CHECK-NEXT: (#\a)
(string->list "a")
;; CHECK-NEXT: (#\a #\b #\c)
(string->list "abc")
;; CHECK-NEXT: (#\H #\e #\l #\l #\o)
(string->list "Hello")
;; CHECK-NEXT: (#\1 #\2 #\3)
(string->list "123")
;; CHECK-NEXT: (#\space #\ht #\newline)
(string->list " \t\n")

;; Test string with various escape sequences
;; CHECK-NEXT: (#\a #\newline #\b)
(string->list "a\nb")
;; CHECK-NEXT: (#\a #\ht #\b)
(string->list "a\tb")
;; CHECK-NEXT: (#\a #\space #\b)
(string->list "a b")
;; CHECK-NEXT: (#\a #\newline #\ht #\space #\b)
(string->list "a\n\t b")
;; CHECK-NEXT: (#\newline #\ht #\space)
(string->list "\n\t ")
;; CHECK-NEXT: (#\space #\ht #\newline #\space)
(string->list " \t\n ")

;; Test list->string
;; CHECK: "1\""
(list->string '(#\1 #\\ #\"))
;; CHECK-NEXT: ""
(list->string '())
;; CHECK-NEXT: "a"
(list->string '(#\a))
;; CHECK-NEXT: "abc"
(list->string '(#\a #\b #\c))
;; CHECK-NEXT: "Hello"
(list->string '(#\H #\e #\l #\l #\o))
;; CHECK-NEXT: "123"
(list->string '(#\1 #\2 #\3))
;; CHECK-NEXT: "P l"
(list->string '(#\P #\space #\l))

;; Test round-trip conversion
;; CHECK: "abc"
(list->string (string->list "abc"))
;; CHECK-NEXT: "Hello, World!"
(list->string (string->list "Hello, World!"))
;; CHECK-NEXT: "123456"
(list->string '(#\1 #\2 #\3 #\4 #\5 #\6))
;; CHECK-NEXT: (#\1 #\2 #\3 #\4 #\5 #\6)
(string->list (list->string '(#\1 #\2 #\3 #\4 #\5 #\6)))

