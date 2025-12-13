;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test member function (uses equal? for comparison)

;; Test member with simple symbol match
;; CHECK: (b c)
(member 'b '(a b c))

;; Test member with symbol not found
;; CHECK: #f
(member 'd '(a b c))

;; Test member with list element (deep comparison)
;; CHECK: ((a) c)
(member (list 'a) '(b (a) c))

;; Test member with list element not found (different structure)
;; CHECK: #f
(member (list 'a 'b) '(b (a) c))

;; Test member with number
;; CHECK: (2 3)
(member 2 '(1 2 3))

;; Test member with number not found
;; CHECK: #f
(member 4 '(1 2 3))

;; Test member with nested lists
;; CHECK: ((1 2) (3 4))
(member '(1 2) '((0 0) (1 2) (3 4)))

;; Test member with empty list
;; CHECK: #f
(member 'a '())

;; Test member with string
;; CHECK: ("world" "!")
(member "world" '("hello" "world" "!"))

;; Test member with character
;; CHECK: (#\b #\c)
(member #\b '(#\a #\b #\c))

;; Test member with boolean
;; CHECK: (#t #f)
(member #t '(#f #t #f))

;; Test member with mixed types
;; CHECK: (2 "test" #t)
(member 2 '(1 2 "test" #t))

;; Test member with duplicate elements (should return first match)
;; CHECK: (1 2 1)
(member 1 '(0 1 2 1))

;; Test memv function (uses eqv? for comparison)

;; Test memv with number (eqv? compares by value)
;; CHECK: (101 102)
(memv 101 '(100 101 102))

;; Test memv with number not found
;; CHECK: #f
(memv 103 '(100 101 102))

;; Test memv with character
;; CHECK: (#\b #\c)
(memv #\b '(#\a #\b #\c))

;; Test memv with boolean
;; CHECK: (#t #f)
(memv #t '(#f #t #f))

;; Test memv with symbol (should use eq? comparison, so new list won't match)
;; CHECK: #f
(memv (list 'a) '(b (a) c))

;; Test memv with empty list
;; CHECK: #f
(memv 'a '())

;; Test memv with duplicate numbers (should return first match)
;; CHECK: (2 3 2)
(memv 2 '(1 2 3 2))

;; Test memv with float numbers
;; CHECK: (2.5 3.0)
(memv 2.5 '(1.0 2.5 3.0))

;; Test memv with mixed numeric types
;; CHECK: (2 3)
(memv 2 '(1 2 3))

