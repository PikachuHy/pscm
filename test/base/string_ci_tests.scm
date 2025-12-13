;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test case-insensitive string comparison functions

;; Test string-ci=? (case-insensitive string equality)
;; CHECK: #t
(string-ci=? "" "")
;; CHECK-NEXT: #f
(string-ci=? "A" "B")
;; CHECK-NEXT: #f
(string-ci=? "a" "B")
;; CHECK-NEXT: #f
(string-ci=? "A" "b")
;; CHECK-NEXT: #f
(string-ci=? "a" "b")
;; CHECK-NEXT: #f
(string-ci=? "9" "0")
;; CHECK-NEXT: #t
(string-ci=? "A" "A")
;; CHECK-NEXT: #t
(string-ci=? "A" "a")
;; CHECK-NEXT: #t
(string-ci=? "a" "A")
;; CHECK-NEXT: #t
(string-ci=? "Hello" "HELLO")
;; CHECK-NEXT: #t
(string-ci=? "HELLO" "hello")
;; CHECK-NEXT: #t
(string-ci=? "Hello" "hElLo")

;; Test string-ci<? (case-insensitive string less than)
;; CHECK: #f
(string-ci<? "" "")
;; CHECK-NEXT: #t
(string-ci<? "A" "B")
;; CHECK-NEXT: #t
(string-ci<? "a" "B")
;; CHECK-NEXT: #t
(string-ci<? "A" "b")
;; CHECK-NEXT: #t
(string-ci<? "a" "b")
;; CHECK-NEXT: #f
(string-ci<? "9" "0")
;; CHECK-NEXT: #f
(string-ci<? "A" "A")
;; CHECK-NEXT: #f
(string-ci<? "A" "a")
;; CHECK-NEXT: #f
(string-ci<? "a" "A")
;; CHECK-NEXT: #t
(string-ci<? "aa" "aaA")
;; CHECK-NEXT: #t
(string-ci<? "aa" "AAA")
;; CHECK-NEXT: #t
(string-ci<? "aA" "Aaa")
;; CHECK-NEXT: #f
(string-ci<? "AAA" "Aaa")

;; Test string-ci>? (case-insensitive string greater than)
;; CHECK: #f
(string-ci>? "" "")
;; CHECK-NEXT: #f
(string-ci>? "A" "B")
;; CHECK-NEXT: #f
(string-ci>? "a" "B")
;; CHECK-NEXT: #f
(string-ci>? "A" "b")
;; CHECK-NEXT: #f
(string-ci>? "a" "b")
;; CHECK-NEXT: #t
(string-ci>? "9" "0")
;; CHECK-NEXT: #f
(string-ci>? "A" "A")
;; CHECK-NEXT: #f
(string-ci>? "A" "a")
;; CHECK-NEXT: #f
(string-ci>? "a" "A")
;; CHECK-NEXT: #t
(string-ci>? "aaA" "aa")
;; CHECK-NEXT: #t
(string-ci>? "AAa" "aa")
;; CHECK-NEXT: #t
(string-ci>? "AaA" "aA")
;; CHECK-NEXT: #t
(string-ci>? "BB" "aaa")

;; Test string-ci<=? (case-insensitive string less than or equal)
;; CHECK: #t
(string-ci<=? "" "")
;; CHECK-NEXT: #t
(string-ci<=? "A" "B")
;; CHECK-NEXT: #t
(string-ci<=? "a" "B")
;; CHECK-NEXT: #t
(string-ci<=? "A" "b")
;; CHECK-NEXT: #t
(string-ci<=? "a" "b")
;; CHECK-NEXT: #f
(string-ci<=? "9" "0")
;; CHECK-NEXT: #t
(string-ci<=? "A" "A")
;; CHECK-NEXT: #t
(string-ci<=? "A" "a")
;; CHECK-NEXT: #t
(string-ci<=? "a" "A")
;; CHECK-NEXT: #t
(string-ci<=? "aa" "AA")
;; CHECK-NEXT: #t
(string-ci<=? "aa" "aaA")
;; CHECK-NEXT: #t
(string-ci<=? "aa" "AAA")
;; CHECK-NEXT: #t
(string-ci<=? "aA" "Aaa")

;; Test string-ci>=? (case-insensitive string greater than or equal)
;; CHECK: #t
(string-ci>=? "" "")
;; CHECK-NEXT: #f
(string-ci>=? "A" "B")
;; CHECK-NEXT: #f
(string-ci>=? "a" "B")
;; CHECK-NEXT: #f
(string-ci>=? "A" "b")
;; CHECK-NEXT: #f
(string-ci>=? "a" "b")
;; CHECK-NEXT: #t
(string-ci>=? "9" "0")
;; CHECK-NEXT: #t
(string-ci>=? "A" "A")
;; CHECK-NEXT: #t
(string-ci>=? "A" "a")
;; CHECK-NEXT: #t
(string-ci>=? "a" "A")
;; CHECK-NEXT: #t
(string-ci>=? "aa" "AA")
;; CHECK-NEXT: #f
(string-ci>=? "aa" "aaA")
;; CHECK-NEXT: #f
(string-ci>=? "aa" "AAA")
;; CHECK-NEXT: #f
(string-ci>=? "aA" "Aaa")

;; Test edge cases with different lengths
;; CHECK: #t
(string-ci<? "a" "AA")
;; CHECK-NEXT: #f
(string-ci<? "AA" "a")
;; CHECK-NEXT: #t
(string-ci>? "AA" "a")
;; CHECK-NEXT: #f
(string-ci>? "a" "AA")
;; CHECK-NEXT: #t
(string-ci<=? "a" "AA")
;; CHECK-NEXT: #f
(string-ci<=? "AA" "a")
;; CHECK-NEXT: #t
(string-ci>=? "AA" "a")
;; CHECK-NEXT: #f
(string-ci>=? "a" "AA")

