;; RUN: %pscm_cc --test %s | FileCheck %s

(define v " abc ")
;; CHECK: " abc "
v
;; CHECK: " abc "
(do ((i (- (string-length v) 1) (- i 1)))
  ((< i 0) v)
  #t)
