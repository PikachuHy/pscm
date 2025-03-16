;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; RUN: %pscm_main --test %s | FileCheck %s

(define-macro (a b) `(+ 1 ,b))
;; CHECK: 3
(a 2)

(define (str-copy v)
    (do ((i (- 2 1) (- i 1)))
        ((< i 0) v)
        (+ 1)))

(define-macro (do bindings test-and-result . body)
        (let ((variables (map car bindings))
              (inits     (map cadr bindings))
              (steps     (map (lambda (clause)
                                (if (null? (cddr clause))
                                    (car clause)
                                    (caddr clause)))
                              bindings))
              (test   (car test-and-result))
              (result (cdr test-and-result))
              (loop   (gensym)))
  
          `(letrec ((,loop (lambda ,variables
                             (if ,test
                                 ,(if (not (null? result))
                                      `(begin . ,result))
                                 (begin
                                   ,@body
                                   (,loop . ,steps))))))
             (,loop . ,inits)) ))
;; CHECK: "a"
(str-copy "a")
;; CHECK: "a"
(str-copy "a")

