;; RUN: %pscm_cc --test %s | FileCheck %s

(define inp (current-input-port))
;; CHECK: #t
(input-port? inp)

(define out (current-output-port))
;; CHECK: #t
(output-port? out)

;; Verify set-current-output-port returns the old port
(define old-out (set-current-output-port out))
;; CHECK: #t
(output-port? old-out)
