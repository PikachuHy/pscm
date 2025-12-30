;; RUN: %pscm_cc --test %s | FileCheck %s
;; Test port and throw operations (compatible with Guile 1.8)

;; Test 1: current-error-port returns a port
;; CHECK: #t
(output-port? (current-error-port))

;; Test 2: force-output (no arguments, flushes stdout)
(display "test")
(newline)
(force-output)
;; CHECK-NEXT: test

;; Test 3: force-output with port argument
(define out-port (open-output-string))
(display "hello" out-port)
(force-output out-port)
;; CHECK-NEXT: #t
(output-port? out-port)

;; Test 4: set-current-error-port
(define old-err-port (current-error-port))
(define new-err-port (open-output-string))
(define returned-port (set-current-error-port new-err-port))
;; CHECK-NEXT: #t
(eq? returned-port old-err-port)

;; CHECK-NEXT: #t
(eq? (current-error-port) new-err-port)

;; Restore old error port
(set-current-error-port old-err-port)

;; Test 5: current-error-port is used by error messages
;; This is tested indirectly through catch/throw tests
