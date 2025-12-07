;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; RUN: %pscm_main --test %s | FileCheck %s
;; RUN: %pscm_cc --test %s | FileCheck %s

(define v " abc ")
;; CHECK: " abc "
v
;; CHECK: " abc "
(do ((i (- (string-length v) 1) (- i 1)))
  ((< i 0) v)
  #t)
