;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test basic quasiquote
;; CHECK: (a b c)
`(a b c)

;; Test quasiquote with unquote
(define x 42)
;; CHECK: (a 42 c)
`(a ,x c)

;; Test quasiquote with multiple unquotes
(define y 100)
;; CHECK: (a 42 b 100 c)
`(a ,x b ,y c)

;; Test quasiquote in macro
(define-macro (add-one x) `(+ 1 ,x))
;; CHECK: 4
(add-one 3)

;; Test quasiquote with numbers and symbols
;; CHECK: (1 2 3)
`(1 2 3)

;; Test quasiquote with mixed content
(define a 10)
(define b 20)
;; CHECK: (10 x 20 y)
`(,a x ,b y)

