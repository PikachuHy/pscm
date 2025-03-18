;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; TODO: %pscm_main --test %s | FileCheck %s

;;; Test code from https://www.shido.info/lisp/scheme_cc_e.html 4.3. Coroutine

(define (1+ x) (+ x 1))
;;; queue
(define (make-queue)
  (cons '() '()))

(define (enqueue! queue obj)
  (let ((lobj (list obj)))
    (if (null? (car queue))
	(begin
	  (set-car! queue lobj)
	  (set-cdr! queue lobj))
	(begin
	  (set-cdr! (cdr queue) lobj)
	  (set-cdr! queue lobj)))
    (car queue)))

(define (dequeue! queue)
  (let ((obj (car (car queue))))
    (set-car! queue (cdr (car queue)))
 obj))


;;; coroutine   
(define process-queue (make-queue))

(define (coroutine thunk)
  (enqueue! process-queue thunk))

(define (start)
   ((dequeue! process-queue)))
   
(define (pause)
  (call/cc
   (lambda (k)
     (coroutine (lambda () (k #f)))
     (start))))


;;; example
;; CHECK: (#<procedure #f ()>)
(coroutine (lambda ()
	     (let loop ((i 0)) 
	       (if (< i 10)
		   (begin
		     (display (1+ i)) 
		     (display " ") 
		     (pause) 
		     (loop (1+ i)))))))
     		   
;; CHECK: (#<procedure #f ()> #<procedure #f ()>)
(coroutine (lambda ()
	     (let loop ((i 0)) 
	       (if (< i 10)
		   (begin
		     (display (integer->char (+ i 97)))
		     (display " ")
		     (pause) 
		     (loop (1+ i)))))))

(newline)
;; CHECK: 1 a 2 b 3 c 4 d 5 e 6 f 7 g 8 h 9 i 10 j
(start)
