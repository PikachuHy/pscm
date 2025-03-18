;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; TODO: %pscm_main --test %s | FileCheck %s

(define (find-leaf obj tree)
  (call/cc
   (lambda (cc)
     (letrec ((iter
	       (lambda (tree)
		 (cond
		  ((null?  tree) #f)
		  ((pair? tree)
		   (iter (car tree))
		   (iter (cdr tree)))
		  (else
		   (if (eqv? obj tree)
		       (cc obj)))))))
       (iter tree)))))

;; CHECK: 7
(find-leaf 7 '(1 (2 3) 4 (5 (6 7))))
;; CHECK: #f
(find-leaf 8 '(1 (2 3) 4 (5 (6 7))))
