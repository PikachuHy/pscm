;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test case-insensitive character comparison functions

;; Test char-ci=? (case-insensitive character equality)
;; CHECK: #t
(char-ci=? #\A #\A)
;; CHECK-NEXT: #t
(char-ci=? #\A #\a)
;; CHECK-NEXT: #t
(char-ci=? #\a #\A)
;; CHECK-NEXT: #t
(char-ci=? #\a #\a)
;; CHECK-NEXT: #f
(char-ci=? #\A #\B)
;; CHECK-NEXT: #f
(char-ci=? #\a #\B)
;; CHECK-NEXT: #f
(char-ci=? #\A #\b)
;; CHECK-NEXT: #f
(char-ci=? #\a #\b)
;; CHECK-NEXT: #f
(char-ci=? #\9 #\0)
;; CHECK-NEXT: #t
(char-ci=? #\Z #\z)
;; CHECK-NEXT: #t
(char-ci=? #\z #\Z)

;; Test char-ci<? (case-insensitive character less than)
;; CHECK: #t
(char-ci<? #\A #\B)
;; CHECK-NEXT: #t
(char-ci<? #\a #\B)
;; CHECK-NEXT: #t
(char-ci<? #\A #\b)
;; CHECK-NEXT: #t
(char-ci<? #\a #\b)
;; CHECK-NEXT: #f
(char-ci<? #\9 #\0)
;; CHECK-NEXT: #f
(char-ci<? #\A #\A)
;; CHECK-NEXT: #f
(char-ci<? #\A #\a)
;; CHECK-NEXT: #f
(char-ci<? #\a #\A)
;; CHECK-NEXT: #t
(char-ci<? #\0 #\A)
;; CHECK-NEXT: #t
(char-ci<? #\A #\z)

;; Test char-ci>? (case-insensitive character greater than)
;; CHECK: #f
(char-ci>? #\A #\B)
;; CHECK-NEXT: #f
(char-ci>? #\a #\B)
;; CHECK-NEXT: #f
(char-ci>? #\A #\b)
;; CHECK-NEXT: #f
(char-ci>? #\a #\b)
;; CHECK-NEXT: #t
(char-ci>? #\9 #\0)
;; CHECK-NEXT: #f
(char-ci>? #\A #\A)
;; CHECK-NEXT: #f
(char-ci>? #\A #\a)
;; CHECK-NEXT: #f
(char-ci>? #\a #\A)
;; CHECK-NEXT: #t
(char-ci>? #\z #\A)
;; CHECK-NEXT: #t
(char-ci>? #\Z #\a)

;; Test char-ci<=? (case-insensitive character less than or equal)
;; CHECK: #t
(char-ci<=? #\A #\B)
;; CHECK-NEXT: #t
(char-ci<=? #\a #\B)
;; CHECK-NEXT: #t
(char-ci<=? #\A #\b)
;; CHECK-NEXT: #t
(char-ci<=? #\a #\b)
;; CHECK-NEXT: #f
(char-ci<=? #\9 #\0)
;; CHECK-NEXT: #t
(char-ci<=? #\A #\A)
;; CHECK-NEXT: #t
(char-ci<=? #\A #\a)
;; CHECK-NEXT: #t
(char-ci<=? #\a #\A)
;; CHECK-NEXT: #t
(char-ci<=? #\z #\Z)

;; Test char-ci>=? (case-insensitive character greater than or equal)
;; CHECK: #f
(char-ci>=? #\A #\B)
;; CHECK-NEXT: #f
(char-ci>=? #\a #\B)
;; CHECK-NEXT: #f
(char-ci>=? #\A #\b)
;; CHECK-NEXT: #f
(char-ci>=? #\a #\b)
;; CHECK-NEXT: #t
(char-ci>=? #\9 #\0)
;; CHECK-NEXT: #t
(char-ci>=? #\A #\A)
;; CHECK-NEXT: #t
(char-ci>=? #\A #\a)
;; CHECK-NEXT: #t
(char-ci>=? #\a #\A)
;; CHECK-NEXT: #t
(char-ci>=? #\Z #\z)

