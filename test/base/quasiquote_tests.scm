;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test parsing of quasiquote expressions
;; These tests verify that the parser correctly handles:
;; 1. Quoted unquote in lists: ',name
;; 2. Unquote in vectors: #(...,expr,...)
;; 3. Direct unquote after quasiquote: `,expr

;; Test basic quasiquote
;; CHECK: (quasiquote (list a b))
'`(list a b)

;; Test unquote in list
;; CHECK: (quasiquote (list (unquote name) b))
'`(list ,name b)

;; Test quoted unquote in list (',name syntax)
;; CHECK: (quasiquote (list (unquote name) '(unquote name)))
'`(list ,name ',name)

;; Test unquote-splicing in list
;; CHECK: (quasiquote (list (unquote name) (unquote-splicing items) b))
'`(list ,name ,@items b)

;; Test unquote in vector
;; CHECK: (quasiquote #(10 5 (unquote (+ 1 1)) 8))
'`#(10 5 ,(+ 1 1) 8)

;; Test unquote-splicing in vector
;; CHECK: (quasiquote #(10 5 (unquote (sqrt 4)) (unquote-splicing (map sqrt '(16 9))) 8))
'`#(10 5 ,(sqrt 4) ,@(map sqrt '(16 9)) 8)

;; Test direct unquote after quasiquote
;; CHECK: (quasiquote (unquote (+ 2 3)))
'`,(+ 2 3)

;; Test nested quasiquote
;; CHECK: (quasiquote (list (quasiquote (unquote x)) (unquote y)))
'`(list `,x ,y)

;; Test quote inside quasiquote
;; CHECK: (quasiquote (list 'a (unquote name)))
'`(list 'a ,name)

;; Test complex example with multiple unquotes
;; CHECK: (quasiquote (list (unquote a) (unquote b) '(unquote c) d))
'`(list ,a ,b ',c d)

;; Test unquote with expression
;; CHECK: (quasiquote (list (unquote (+ 1 2)) 4))
'`(list ,(+ 1 2) 4)

;; Test empty quasiquote list
;; CHECK: (quasiquote ())
'`()

;; Test quasiquote with only unquote
;; CHECK: (quasiquote ((unquote expr)))
'`(,expr)

;; Test quasiquote with only quoted unquote
;; CHECK: (quasiquote ('(unquote expr)))
'`(',expr)

;; Test nested unquote (double comma: ,,name means (unquote (unquote name)))
;; CHECK: (quasiquote (unquote (unquote name1)))
'`,,name1

;; Test nested quasiquote with nested unquote
;; CHECK: (quasiquote (a (quasiquote (b (unquote (unquote name1)) (unquote '(unquote name2)) d)) e))
'`(a `(b ,,name1 ,',name2 d) e)

;; Test triple nested unquote (three commas)
;; CHECK: (quasiquote (unquote (unquote (unquote name))))
'`,,,name

;; Test nested quasiquote with triple nested unquote
;; CHECK: (quasiquote (a (quasiquote (b (quasiquote (c (unquote (unquote (unquote name))) d)) e)) f))
'`(a `(b `(c ,,,name d) e) f)

;; Test nested unquote with expression
;; CHECK: (quasiquote (unquote (unquote (+ 1 2))))
'`,,(+ 1 2)

;; Test nested unquote-splicing (not valid in standard Scheme, but test parsing)
;; CHECK: (quasiquote (unquote (unquote-splicing items)))
'`,,@items

;; ========================================
;; Test actual evaluation of quasiquote expressions
;; ========================================

;; Test basic quasiquote evaluation
;; CHECK: (a b c)
`(a b c)

;; Test unquote with simple value
;; CHECK: (a 42 c)
`(a ,(+ 20 22) c)

;; Test unquote with variable
(define x 10)
;; CHECK: (a 10 c)
`(a ,x c)

;; Test unquote-splicing with empty list
;; CHECK: (a c)
`(a ,@'() c)

;; Test unquote-splicing with single element list
;; CHECK: (a 1 c)
`(a ,@'(1) c)

;; Test unquote-splicing with multiple elements (the bug we just fixed)
;; CHECK: (a 1 2 3 c)
`(a ,@'(1 2 3) c)

;; Test unquote-splicing with computed list
;; CHECK: (a 1 2 3 c)
`(a ,@(list 1 2 3) c)

;; Test unquote-splicing with map result (the specific bug case)
;; CHECK: (a 3 4 5 6 b)
`(a ,(+ 1 2) ,@(map abs '(4 -5 6)) b)

;; Test unquote-splicing at the beginning
;; CHECK: (1 2 3 a b)
`(,@'(1 2 3) a b)

;; Test unquote-splicing at the end
;; CHECK: (a b 1 2 3)
`(a b ,@'(1 2 3))

;; Test multiple unquote-splicing
;; CHECK: (a 1 2 b 3 4 c)
`(a ,@'(1 2) b ,@'(3 4) c)

;; Test unquote and unquote-splicing together
(define name 'foo)
(define items '(x y z))
;; CHECK: (list foo x y z bar)
`(list ,name ,@items bar)

;; Test unquote-splicing with empty result (using map that returns empty list)
;; CHECK: (a c)
`(a ,@'() c)

;; Test nested quasiquote (currently preserved as structure, not evaluated)
(define y 20)
;; CHECK: (a (quasiquote b (unquote . y) d) e)
`(a `(b ,y d) e)

;; Test unquote with expression that returns list
;; CHECK: ((1 2 3) 4 5)
`(,(list 1 2 3) 4 5)

;; Test unquote-splicing with expression that returns list
;; CHECK: (1 2 3 4 5)
`(,@(list 1 2 3) 4 5)

;; Test complex example: building a function call
(define func 'list)
(define args '(1 2 3))
;; CHECK: (list 1 2 3)
`(,func ,@args)

;; Test unquote-splicing with map and filter
;; CHECK: (a 2 4 6 b)
`(a ,@(map (lambda (x) (* x 2)) '(1 2 3)) b)

;; Test unquote-splicing with computed list
;; CHECK: (a 2 4 6 b)
`(a ,@(map (lambda (x) (* x 2)) '(1 2 3)) b)

;; Test empty list with unquote-splicing
;; CHECK: (a b)
`(a ,@'() b)

;; Test unquote-splicing with single element
;; CHECK: (a 42 b)
`(a ,@'(42) b)

;; Test R4RS test case: list with unquote
;; CHECK: (list 3 4)
`(list ,(+ 1 2) 4)

;; Test R4RS test case: list with variable and quoted variable
(let ((name 'a))
  ;; CHECK: (list a (quote a))
  `(list ,name ',name))

;; Test R4RS test case: complex unquote-splicing
;; CHECK: ((foo 7) . cons)
`((foo ,(- 10 3)) ,@(cdr '(c)) . ,(car '(cons)))
