;; TODO: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; RUN: %pscm_main --test %s | FileCheck %s

(define h (make-hash-table 31))
;; CHECK: #<hash-table 0/31
h
;; CHECK: "bar"
(hash-set! h 'foo "bar")
;; CHECK: "bar"
(hash-ref h 'foo)

;; CHECK: "bar2"
(hash-set! h :check-mark "bar2")
;; CHECK: "bar2"
(hash-ref h :check-mark)

(define h (make-hash-table 31))
(hash-set! h 'foo "bar")
(hash-set! h 'braz "zonk")
;; CHECK: 2
(hash-fold (lambda (key value seed) (+ 1 seed)) 0 h)
;; CHECk: (frob . #f)
(hashq-create-handle! h 'frob #f)
;; CHECK: (foo . "bar")
(hashq-get-handle h 'foo)
;; CHECK: #f
(hashq-get-handle h 'not-there)
;; CHECK: 3
(hash-fold (lambda (key value seed) (+ 1 seed)) 0 h)
(hash-remove! h 'foo)
;; CHECK: #f
(hash-ref h 'foo)
;; CHECK: 2
(hash-fold (lambda (key value seed) (+ 1 seed)) 0 h)


(define (ahash-table->list h)
	(hash-fold acons '() (car h)))
(define h (make-hash-table 31))
(hash-set! h 'foo "bar")
(hash-set! h 'braz "zonk")
;; CHECK: ((braz . "zonk") (foo . "bar"))
(ahash-table->list (cons h '()))
