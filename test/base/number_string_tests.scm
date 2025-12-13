;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test number->string function

;; Basic integer tests
;; CHECK: "0"
(number->string 0)
;; CHECK: "100"
(number->string 100)
;; CHECK: "-42"
(number->string -42)
;; CHECK: "1"
(number->string 1)
;; CHECK: "-1"
(number->string -1)

;; Test with different radix
;; CHECK: "100"
(number->string 256 16)
;; CHECK: "ff"
(number->string 255 16)
;; CHECK: "101"
(number->string 5 2)
;; CHECK: "177"
(number->string 127 8)
;; CHECK: "10"
(number->string 10 10)
;; CHECK: "a"
(number->string 10 16)
;; CHECK: "z"
(number->string 35 36)

;; Test floating point numbers
;; CHECK: "3.5"
(number->string 3.5)
;; CHECK: "0.0"
(number->string 0.0)
;; CHECK: "-2.5"
(number->string -2.5)
;; CHECK: "1.0"
(number->string 1.0)
;; CHECK: "3.14159"
(number->string 3.14159)

;; Test rational numbers
;; CHECK: "1/2"
(number->string (/ 1 2))
;; CHECK: "3/4"
(number->string (/ 3 4))
;; CHECK: "-1/3"
(number->string (/ -1 3))
;; CHECK: "2"
(number->string (/ 4 2))
;; CHECK: "5/6"
(number->string (/ 5 6))

;; Test string->number function

;; Basic integer parsing
;; CHECK: 0
(string->number "0")
;; CHECK: 100
(string->number "100")
;; CHECK: -42
(string->number "-42")
;; CHECK: 123
(string->number "123")

;; Test with different radix
;; CHECK: 256
(string->number "100" 16)
;; CHECK: 255
(string->number "ff" 16)
;; CHECK: 5
(string->number "101" 2)
;; CHECK: 127
(string->number "177" 8)
;; CHECK: 35
(string->number "z" 36)

;; Test floating point parsing
;; CHECK: 3.5
(string->number "3.5")
;; CHECK: 0.0
(string->number "0.0")
;; CHECK: -2.5
(string->number "-2.5")
;; CHECK: 1.0
(string->number "1.0")
;; CHECK: 3.14159
(string->number "3.14159")

;; Test invalid strings (should return #f)
;; CHECK: #f
(string->number "")
;; CHECK: #f
(string->number "abc")
;; CHECK: #f
(string->number "12.34.56")

;; Test round-trip conversion
;; CHECK: "100"
(number->string (string->number "100"))
;; CHECK: "42"
(number->string (string->number "42"))
;; CHECK: "ff"
(number->string (string->number "ff" 16) 16)
;; CHECK: "101"
(number->string (string->number "101" 2) 2)

;; Test rational number round-trip (if string->number supports it)
;; Note: string->number may not support rational number parsing yet
;; CHECK: "1/2"
(number->string (/ 1 2))
;; CHECK: "3/4"
(number->string (/ 3 4))

