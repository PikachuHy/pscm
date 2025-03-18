;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; TODO: %pscm_main --test %s | FileCheck %s

;; CHECK: #t
(call-with-current-continuation procedure?)

;; CHECK: 7
(call-with-current-continuation (lambda (k) (+ 2 5)))

;; CHECK: 3
(call-with-current-continuation (lambda (k) (+ 2 5 (k 3))))
