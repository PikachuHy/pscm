;; RUN: %pscm_cc --test %s | FileCheck %s


;; CHECK: ;
(integer->char 59)
;; CHECK: 46
(char->integer #\.)
;; CHECK: 65
(char->integer #\A)

;; CHECK: #\.
(integer->char (char->integer #\.))
;; CHECK: #\A
(integer->char (char->integer #\A))
;; CHECK: #\a
(integer->char (char->integer #\a))

;; CHECK: .
#\.

;; CHECK: (test #\. integer->char (char->integer #\.))
`(test #\. integer->char (char->integer #\.))

;; Test char-upcase and char-downcase
;; CHECK: #\A
(char-upcase #\a)
;; CHECK: #\a
(char-downcase #\A)
;; CHECK: #\Z
(char-upcase #\z)
;; CHECK: #\z
(char-downcase #\Z)
;; CHECK: #\A
(char-upcase #\A)
;; CHECK: #\a
(char-downcase #\a)
;; CHECK: #\1
(char-upcase #\1)
;; CHECK: #\1
(char-downcase #\1)
;; CHECK: #\!
(char-upcase #\!)
;; CHECK: #\!
(char-downcase #\!)
