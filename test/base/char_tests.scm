;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; RUN: %pscm_main --test %s | FileCheck %s

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
