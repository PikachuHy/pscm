;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test the not function
;; In Scheme, not returns #t if the argument is #f, #f otherwise
;; Only #f is falsy in Scheme, everything else (including nil) is truthy

;; Test not with #f (should return #t)
;; CHECK: #t
(not #f)

;; Test not with #t (should return #f)
;; CHECK: #f
(not #t)

;; Test not with number (should return #f, as numbers are truthy)
;; CHECK: #f
(not 3)

;; Test not with zero (should return #f, as 0 is truthy in Scheme)
;; CHECK: #f
(not 0)

;; Test not with negative number (should return #f)
;; CHECK: #f
(not -1)

;; Test not with empty list (should return #f, as () is truthy in Scheme)
;; CHECK: #f
(not '())

;; Test not with non-empty list (should return #f)
;; CHECK: #f
(not (list 3))

;; Test not with list containing elements (should return #f)
;; CHECK: #f
(not (list 1 2 3))

;; Test not with symbol (should return #f)
;; CHECK: #f
(not 'nil)

;; Test not with string (should return #f)
;; CHECK: #f
(not "hello")

;; Test not with empty string (should return #f)
;; CHECK: #f
(not "")

;; Test not with vector (should return #f)
;; CHECK: #f
(not #(1 2 3))

;; Test not with empty vector (should return #f)
;; CHECK: #f
(not #())

;; Test not with procedure (should return #f)
;; CHECK: #f
(not +)

;; Test not with float (should return #f)
;; CHECK: #f
(not 3.14)

;; Test not with zero float (should return #f)
;; CHECK: #f
(not 0.0)

;; Test nested not (not (not #f) should return #f)
;; CHECK: #f
(not (not #f))

;; Test nested not (not (not #t) should return #t)
;; CHECK: #t
(not (not #t))

;; Test not with quoted #f (should return #t, as '#f evaluates to #f)
;; CHECK: #t
(not '#f)

