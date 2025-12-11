;; RUN: %pscm_cc --test %s | FileCheck %s

;;; dynamic-wind
;;; CHECK: (a b c)
(let* ((path '())
           (add (lambda (s) (set! path (cons s path)))))
      (dynamic-wind (lambda () (add 'a)) (lambda () (add 'b)) (lambda () (add 'c)))
      (reverse path))

;;; CHECK: (connect talk1 disconnect connect talk2 disconnect)
(let ((path '())
          (c #f))
      (let ((add (lambda (s)
                   (set! path (cons s path)))))
        (dynamic-wind
            (lambda () (add 'connect))
            (lambda ()
              (add (call-with-current-continuation
                    (lambda (c0)
                      (set! c c0)
                      'talk1))))
            (lambda () (add 'disconnect)))
        (if (< (length path) 4)
            (c 'talk2)
            (reverse path))))


(define (f c) c)
;;; CHECK: #<continuation
(call/cc f)
;;; CHECK: #<continuation
(call/cc (call/cc f))
