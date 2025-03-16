;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; RUN: %pscm_main --test %s | FileCheck %s

;; TODO: add check

;; CHECK: 46
(for-each (lambda (x y) (display (+ x y)))
                    '(1 2)
                    '(3 4))

(for-each abs '(4 -5 6))


(define (test . args) #t)

(for-each (lambda (x y) (list x y)) (list 'a) (list 'b))

(define (test . args) #t)

(for-each (lambda (x y)
    (for-each (lambda (f)
                      (test #f f x))
              (list null?)))
              (list '()  '(test))
              (list '()  '(t . t)))

(define (test . args) #t)


(for-each (lambda (x y)
    (for-each (lambda (f)
                      (test #f f x))
                      (list null? '(test))))
                      (list '()  '(test))
                      (list '()  '(t . t)))
                      