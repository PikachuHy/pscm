;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test string-append
;; CHECK: "HelloWorld"
(string-append "Hello" "World")

;; CHECK: "abc"
(string-append "a" "b" "c")

;; CHECK: "Hello World"
(string-append "Hello" " " "World")

;; Test string-length
;; CHECK: 5
(string-length "Hello")

;; CHECK: 0
(string-length "")

;; CHECK: 11
(string-length "Hello World")

;; Test string-ref
;; CHECK: #\H
(string-ref "Hello" 0)

;; CHECK: #\o
(string-ref "Hello" 4)

;; CHECK: #\l
(string-ref "Hello" 2)

;; Test substring
;; CHECK: "ell"
(substring "Hello" 1 4)

;; CHECK: "Hello"
(substring "Hello" 0 5)

;; CHECK: "lo"
(substring "Hello" 3 5)

;; Test string operations with edge cases
;; CHECK: ""
(substring "Hello" 0 0)

;; CHECK: "H"
(substring "Hello" 0 1)

;; CHECK: "o"
(substring "Hello" 4 5)

