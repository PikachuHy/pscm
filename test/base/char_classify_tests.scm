;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test character classification functions

;; Test char-alphabetic?
;; CHECK: #t
(char-alphabetic? #\a)
;; CHECK-NEXT: #t
(char-alphabetic? #\A)
;; CHECK-NEXT: #t
(char-alphabetic? #\z)
;; CHECK-NEXT: #t
(char-alphabetic? #\Z)
;; CHECK-NEXT: #f
(char-alphabetic? #\0)
;; CHECK-NEXT: #f
(char-alphabetic? #\9)
;; CHECK-NEXT: #f
(char-alphabetic? #\space)
;; CHECK-NEXT: #f
(char-alphabetic? #\;)
;; CHECK-NEXT: #f
(char-alphabetic? #\newline)
;; CHECK-NEXT: #f
(char-alphabetic? #\tab)

;; Test char-numeric?
;; CHECK: #f
(char-numeric? #\a)
;; CHECK-NEXT: #f
(char-numeric? #\A)
;; CHECK-NEXT: #f
(char-numeric? #\z)
;; CHECK-NEXT: #f
(char-numeric? #\Z)
;; CHECK-NEXT: #t
(char-numeric? #\0)
;; CHECK-NEXT: #t
(char-numeric? #\9)
;; CHECK-NEXT: #f
(char-numeric? #\space)
;; CHECK-NEXT: #f
(char-numeric? #\;)
;; CHECK-NEXT: #f
(char-numeric? #\newline)

;; Test char-whitespace?
;; CHECK: #f
(char-whitespace? #\a)
;; CHECK-NEXT: #f
(char-whitespace? #\A)
;; CHECK-NEXT: #f
(char-whitespace? #\z)
;; CHECK-NEXT: #f
(char-whitespace? #\Z)
;; CHECK-NEXT: #f
(char-whitespace? #\0)
;; CHECK-NEXT: #f
(char-whitespace? #\9)
;; CHECK-NEXT: #t
(char-whitespace? #\space)
;; CHECK-NEXT: #f
(char-whitespace? #\;)
;; CHECK-NEXT: #t
(char-whitespace? #\newline)
;; CHECK-NEXT: #t
(char-whitespace? #\tab)

;; Test char-upper-case?
;; CHECK: #f
(char-upper-case? #\0)
;; CHECK-NEXT: #f
(char-upper-case? #\9)
;; CHECK-NEXT: #f
(char-upper-case? #\space)
;; CHECK-NEXT: #f
(char-upper-case? #\;)
;; CHECK-NEXT: #t
(char-upper-case? #\A)
;; CHECK-NEXT: #t
(char-upper-case? #\Z)
;; CHECK-NEXT: #f
(char-upper-case? #\a)
;; CHECK-NEXT: #f
(char-upper-case? #\z)
;; CHECK-NEXT: #f
(char-upper-case? #\newline)

;; Test char-lower-case?
;; CHECK: #f
(char-lower-case? #\0)
;; CHECK-NEXT: #f
(char-lower-case? #\9)
;; CHECK-NEXT: #f
(char-lower-case? #\space)
;; CHECK-NEXT: #f
(char-lower-case? #\;)
;; CHECK-NEXT: #f
(char-lower-case? #\A)
;; CHECK-NEXT: #f
(char-lower-case? #\Z)
;; CHECK-NEXT: #t
(char-lower-case? #\a)
;; CHECK-NEXT: #t
(char-lower-case? #\z)
;; CHECK-NEXT: #f
(char-lower-case? #\newline)

