;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; RUN: %pscm_main --test %s | FileCheck %s
;; RUN: %pscm_cc --test %s | FileCheck %s

;; no check here ???
(define v " abc ")
(do ((i (- (string-length v) 1) (- i 1)))
  ((< i 0) v)
  #t)
(do ((i (- (string-length v) 1) (- i 1)))
  ((< i 0) v)
  #t)

;; how to check space?
(define s " abc2 ")
;; CHECK: abc2
(let ((v (make-string (string-length s))))
  (do ((i (- (string-length v) 1) (- i 1)))
    ((< i 0) v)

    (string-set! v i (string-ref s i))))
;; CHECK: abc2
(let ((v (make-string (string-length s))))
  (do ((i (- (string-length v) 1) (- i 1)))
    ((< i 0) v)

    (string-set! v i (string-ref s i))))

(define s " abc3 ")
(define (str-copy s)
  (let ((v (make-string (string-length s))))
    (do ((i (- (string-length v) 1) (- i 1)))
      ((< i 0) v)
      (string-set! v i (string-ref s i)))))
;; CHECK: abc3
(str-copy s)
