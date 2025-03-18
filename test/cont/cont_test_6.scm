;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; TODO: %pscm_main --test %s | FileCheck %s

(define (leaf-generator tree)
  (let ((return '()))                                                        ; 1
    (letrec ((continue                                                      ; 2
	      (lambda ()
		(let loop ((tree tree))                                     ; 3
		  (cond                                                     ; 4
		   ((null? tree) 'skip)                                     ; 5
		   ((pair? tree) (loop (car tree)) (loop (cdr tree)))       ; 6
		   (else                                                    ; 7
		    (call/cc (lambda (lap-to-go)                            ; 8
			       (set! continue (lambda () (lap-to-go 'restart)))    ; 9
			       (return tree))))))                      ;10
		(return '()))))                                         ;11
        (lambda ()                                                     ;12
          (call/cc (lambda (where-to-go)                               ;13
                     (set! return where-to-go)                         ;14
                     (continue)))))))
                    
(define tr '((1 2) (3 (4 5))))
(define p (leaf-generator tr))
;; CHECK: 1
(p)
;; CHECK: 2
(p)
;; CHECK: 3
(p)
;; CHECK: 4
(p)
;; CHECK: 5
(p)
;; CHECK: ()
(p)
;; CHECK: ()
(p)