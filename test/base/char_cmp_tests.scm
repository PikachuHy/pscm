;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test character comparison functions

;; Test char=? (character equality)
;; CHECK: #t
(char=? #\a #\a)
;; CHECK-NEXT: #f
(char=? #\a #\b)
;; CHECK-NEXT: #f
(char=? #\A #\a)
;; CHECK-NEXT: #t
(char=? #\space #\space)
;; CHECK-NEXT: #t
(char=? #\Space #\space)
;; CHECK-NEXT: #f
(char=? #\0 #\9)
;; CHECK-NEXT: #t
(char=? #\newline #\newline)
;; CHECK-NEXT: #t
(char=? '#\a #\a)
;; CHECK-NEXT: #t
(char=? '#\A #\A)

;; Test char<? (character less than)
;; CHECK: #t
(char<? #\A #\B)
;; CHECK-NEXT: #t
(char<? #\a #\b)
;; CHECK-NEXT: #f
(char<? #\B #\A)
;; CHECK-NEXT: #f
(char<? #\b #\a)
;; CHECK-NEXT: #f
(char<? #\A #\A)
;; CHECK-NEXT: #t
(char<? #\0 #\9)
;; CHECK-NEXT: #f
(char<? #\9 #\0)
;; CHECK-NEXT: #t
(char<? #\space #\!)
;; CHECK-NEXT: #t
(char<? #\a #\z)

;; Test char>? (character greater than)
;; CHECK: #f
(char>? #\A #\B)
;; CHECK-NEXT: #f
(char>? #\a #\b)
;; CHECK-NEXT: #t
(char>? #\B #\A)
;; CHECK-NEXT: #t
(char>? #\b #\a)
;; CHECK-NEXT: #f
(char>? #\A #\A)
;; CHECK-NEXT: #f
(char>? #\0 #\9)
;; CHECK-NEXT: #t
(char>? #\9 #\0)
;; CHECK-NEXT: #f
(char>? #\space #\!)
;; CHECK-NEXT: #f
(char>? #\a #\z)
;; CHECK-NEXT: #t
(char>? #\z #\a)

;; Test char<=? (character less than or equal)
;; CHECK: #t
(char<=? #\A #\B)
;; CHECK-NEXT: #t
(char<=? #\a #\b)
;; CHECK-NEXT: #f
(char<=? #\B #\A)
;; CHECK-NEXT: #f
(char<=? #\b #\a)
;; CHECK-NEXT: #t
(char<=? #\A #\A)
;; CHECK-NEXT: #t
(char<=? #\0 #\9)
;; CHECK-NEXT: #f
(char<=? #\9 #\0)
;; CHECK-NEXT: #t
(char<=? #\a #\a)
;; CHECK-NEXT: #t
(char<=? #\z #\z)

;; Test char>=? (character greater than or equal)
;; CHECK: #f
(char>=? #\A #\B)
;; CHECK-NEXT: #f
(char>=? #\a #\b)
;; CHECK-NEXT: #t
(char>=? #\B #\A)
;; CHECK-NEXT: #t
(char>=? #\b #\a)
;; CHECK-NEXT: #t
(char>=? #\A #\A)
;; CHECK-NEXT: #f
(char>=? #\0 #\9)
;; CHECK-NEXT: #t
(char>=? #\9 #\0)
;; CHECK-NEXT: #t
(char>=? #\a #\a)
;; CHECK-NEXT: #t
(char>=? #\z #\z)

;; Test with special characters
;; CHECK: #t
(char=? #\( #\()
;; CHECK-NEXT: #f
(char=? #\( #\))
;; CHECK-NEXT: #t
(char<? #\( #\))
;; CHECK-NEXT: #t
(char<? #\+ #\-)
;; CHECK-NEXT: #f
(char>? #\+ #\-)

;; Test with quoted characters
;; CHECK: #t
(char=? '#\a '#\a)
;; CHECK-NEXT: #f
(char=? '#\a '#\b)
;; CHECK-NEXT: #t
(char<? '#\A '#\B)
;; CHECK-NEXT: #t
(char>? '#\B '#\A)
;; CHECK-NEXT: #t
(char<=? '#\A '#\A)
;; CHECK-NEXT: #t
(char>=? '#\A '#\A)

;; Test with named characters
;; CHECK: #t
(char=? #\space #\space)
;; CHECK-NEXT: #t
(char=? #\Space #\space)
;; CHECK-NEXT: #t
(char=? #\newline #\newline)
;; CHECK-NEXT: #t
(char=? #\tab #\tab)
;; CHECK-NEXT: #f
(char<? #\space #\newline)
;; CHECK-NEXT: #f
(char>? #\newline #\space)

;; Test edge cases: ASCII boundaries
;; CHECK: #t
(char<? #\0 #\A)
;; CHECK-NEXT: #t
(char<? #\A #\a)
;; CHECK-NEXT: #t
(char<? #\a #\{)
;; CHECK-NEXT: #t
(char>? #\{ #\a)
;; CHECK-NEXT: #t
(char>? #\a #\A)
;; CHECK-NEXT: #t
(char>? #\A #\0)
