;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test string comparison functions (case-sensitive)

;; Test string<? (string less than)
;; CHECK: #t
(string<? "A" "B")
;; CHECK-NEXT: #t
(string<? "a" "b")
;; CHECK-NEXT: #f
(string<? "9" "0")
;; CHECK-NEXT: #f
(string<? "A" "A")
;; CHECK-NEXT: #f
(string<? "B" "A")
;; CHECK-NEXT: #t
(string<? "aa" "aaa")
;; CHECK-NEXT: #t
(string<? "aab" "bb")
;; CHECK-NEXT: #f
(string<? "bba" "bb")
;; CHECK-NEXT: #t
(string<? "baa" "bb")
;; CHECK-NEXT: #f
(string<? "" "")

;; Test string>? (string greater than)
;; CHECK: #f
(string>? "A" "B")
;; CHECK-NEXT: #f
(string>? "a" "b")
;; CHECK-NEXT: #t
(string>? "9" "0")
;; CHECK-NEXT: #f
(string>? "A" "A")
;; CHECK-NEXT: #t
(string>? "B" "A")
;; CHECK-NEXT: #f
(string>? "aa" "aaa")
;; CHECK-NEXT: #f
(string>? "aab" "bb")
;; CHECK-NEXT: #t
(string>? "bba" "bb")
;; CHECK-NEXT: #f
(string>? "baa" "bb")
;; CHECK-NEXT: #f
(string>? "" "")

;; Test string<=? (string less than or equal)
;; CHECK: #t
(string<=? "A" "B")
;; CHECK-NEXT: #t
(string<=? "a" "b")
;; CHECK-NEXT: #f
(string<=? "9" "0")
;; CHECK-NEXT: #t
(string<=? "A" "A")
;; CHECK-NEXT: #f
(string<=? "B" "A")
;; CHECK-NEXT: #t
(string<=? "aa" "aaa")
;; CHECK-NEXT: #t
(string<=? "aab" "bb")
;; CHECK-NEXT: #f
(string<=? "bba" "bb")
;; CHECK-NEXT: #t
(string<=? "baa" "bb")
;; CHECK-NEXT: #t
(string<=? "" "")

;; Test string>=? (string greater than or equal)
;; CHECK: #f
(string>=? "A" "B")
;; CHECK-NEXT: #f
(string>=? "a" "b")
;; CHECK-NEXT: #t
(string>=? "9" "0")
;; CHECK-NEXT: #t
(string>=? "A" "A")
;; CHECK-NEXT: #t
(string>=? "B" "A")
;; CHECK-NEXT: #f
(string>=? "aa" "aaa")
;; CHECK-NEXT: #f
(string>=? "aab" "bb")
;; CHECK-NEXT: #t
(string>=? "bba" "bb")
;; CHECK-NEXT: #f
(string>=? "baa" "bb")
;; CHECK-NEXT: #t
(string>=? "" "")

;; Test edge cases with different lengths
;; CHECK: #t
(string<? "a" "aa")
;; CHECK-NEXT: #f
(string<? "aa" "a")
;; CHECK-NEXT: #t
(string>? "aa" "a")
;; CHECK-NEXT: #f
(string>? "a" "aa")
;; CHECK-NEXT: #t
(string<=? "a" "aa")
;; CHECK-NEXT: #f
(string<=? "aa" "a")
;; CHECK-NEXT: #t
(string>=? "aa" "a")
;; CHECK-NEXT: #f
(string>=? "a" "aa")

