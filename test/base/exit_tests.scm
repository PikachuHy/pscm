;; RUN: cd $(dirname %pscm_cc) && %pscm_cc --test %s | FileCheck %s

;; Test that exit function exists and is a procedure
;; CHECK: #t
(procedure? exit)

;; Test exit function registration
;; CHECK: exit function registered successfully
(if (procedure? exit)
    (display "exit function registered successfully\n")
    (display "exit function registration failed\n"))

;; Test that exit is callable
;; CHECK: exit is callable
(if (procedure? exit)
    (display "exit is callable\n")
    (display "exit is not callable\n"))

;; Test exit function type
;; CHECK: #t
(let ((exit-proc exit))
  (procedure? exit-proc))

;; Test that exit can be stored in a variable
;; CHECK: exit stored successfully
(let ((my-exit exit))
  (if (procedure? my-exit)
      (display "exit stored successfully\n")
      (display "exit storage failed\n")))

;; Test exit in a list (should be a procedure)
;; CHECK: (#<builtin-func exit>)
(list exit)

;; Test exit with apply (without actually applying it)
;; CHECK: exit can be used with apply
(if (procedure? exit)
    (display "exit can be used with apply\n")
    (display "exit cannot be used with apply\n"))

;; Test exit function in different contexts
;; CHECK: exit works in let binding
(let ((exit-fn exit))
  (if (procedure? exit-fn)
      (display "exit works in let binding\n")
      (display "exit fails in let binding\n")))

;; Test exit as a value
;; CHECK: exit is a valid value
(if exit
    (display "exit is a valid value\n")
    (display "exit is not a valid value\n"))

;; Test exit can be passed as argument (without calling it)
;; CHECK: exit can be passed as argument
(define (test-proc proc)
  (if (procedure? proc)
      (display "exit can be passed as argument\n")
      (display "exit cannot be passed as argument\n")))
(test-proc exit)

;; Test exit in a lambda closure
;; CHECK: exit works in lambda closure
((lambda (fn)
   (if (procedure? fn)
       (display "exit works in lambda closure\n")
       (display "exit fails in lambda closure\n")))
 exit)

;; Test exit with map (without actually calling it)
;; CHECK: (#<builtin-func exit>)
(map (lambda (x) x) (list exit))

;; Test exit function identity
;; CHECK: #t
(eq? exit exit)

;; Test exit is not nil
;; CHECK: #f
(null? exit)

;; Test exit is not a number
;; CHECK: #f
(number? exit)

;; Test exit is not a string
;; CHECK: #f
(string? exit)

;; Test exit is not a symbol
;; CHECK: #f
(symbol? exit)

;; Test exit is a procedure (redundant but explicit)
;; CHECK: #t
(and (procedure? exit) #t)
