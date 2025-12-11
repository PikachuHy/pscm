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
