;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; RUN: %pscm_cc --test %s | FileCheck %s --check-prefix=LONGJMP
;; RUN: %pscm_cc --test %s | FileCheck %s
;; TODO: %pscm_main --test %s | FileCheck %s

;; LONGJMP-NOT: not supported

(define the-continuation #f)
 
(define (test)
  (let ((i 0))
    ; call/cc calls its first function argument, passing
    ; a continuation variable representing this point in
    ; the program as the argument to that function.
    ;
    ; In this case, the function argument assigns that
    ; continuation to the variable the-continuation.
    ;
    (call/cc (lambda (k) (set! the-continuation k)))
    ;
    ; The next time the-continuation is called, we start here.
    (set! i (+ i 1))
    i))

;; CHECK: 1
(test)
;; CHECK: 2
(the-continuation)
;; CHECK: 3
(the-continuation)
; stores the current continuation (which will print 4 next) away
(define another-continuation the-continuation)

;; CHECK: 1
(test) ; resets the-continuation
;; CHECK: 2
(the-continuation)
;; CHECK: 4
(another-continuation) ; uses the previously stored continuation
