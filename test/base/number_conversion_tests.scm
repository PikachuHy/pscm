;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test number->string
;; CHECK: "42"
(number->string 42)

;; CHECK: "-10"
(number->string -10)

;; CHECK: "0"
(number->string 0)

;; Test string->number
;; CHECK: 42
(string->number "42")

;; CHECK: -10
(string->number "-10")

;; CHECK: 0
(string->number "0")

;; Test number conversion edge cases
;; CHECK: "100"
(number->string 100)

;; Test abs function
;; CHECK: 42
(abs 42)

;; CHECK: 42
(abs -42)

;; CHECK: 0
(abs 0)

;; Test max and min
;; CHECK: 10
(max 1 2 3 10 5)

;; CHECK: 1
(min 1 2 3 10 5)

;; CHECK: 5
(max 5)

;; CHECK: 5
(min 5)

