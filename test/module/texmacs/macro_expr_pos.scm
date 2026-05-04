;; Test: macro object in expression position should be expanded, not errored
;; Exercises eval.cc macro expansion path (line 321 + expand_macros in macro.cc)

(define-macro (wrapper . body)
  `(begin ,@body))

(define-macro (for-each-2 proc lst)
  `(for-each ,proc ,lst))

;; When `wrapper` expands, the evaluator encounters `for-each-2` as a
;; MACRO object in expression position. It must expand it rather than error.
(define result '())
(wrapper
  (for-each-2 (lambda (x) (set! result (cons x result))) '(1 2 3)))
(display (equal? (reverse result) '(1 2 3))) (newline)

;; Test: macro that expands to a self-evaluating value
(define-macro (answer) 42)
(display (wrapper (answer))) (newline)
