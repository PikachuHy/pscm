;; RUN: %pscm_cc --test %s | FileCheck %s
;; Test make-soft-port functionality

;; Test 1: Create a read-write soft port
(define p (make-soft-port
           (vector
            (lambda (c) (display c))
            (lambda (s) (display s))
            (lambda () #t)
            (lambda () #\a)
            (lambda () #t))
           "rw"))
p
;; CHECK: #<input port>

;; Test 2: Write to soft port (output goes to stdout, not captured in test)
;; Note: write-char returns none, but in --test mode it's not printed, so we skip the check
(write-char #\X p)

;; Test 3: Read from soft port
(read-char p)
;; CHECK-NEXT: #\a

;; Test 4: Create output-only soft port
(define p-out (make-soft-port
               (vector
                (lambda (c) (display c))
                (lambda (s) (display s))
                (lambda () #t)
                (lambda () #f)
                (lambda () #t))
               "w"))
(output-port? p-out)
;; CHECK-NEXT: #t

;; Test 5: Create input-only soft port
(define p-in (make-soft-port
              (vector
               (lambda (c) #t)
               (lambda (s) #t)
               (lambda () #t)
               (lambda () #\b)
               (lambda () #t))
              "r"))
(input-port? p-in)
;; CHECK-NEXT: #t

;; Test 6: Read from input-only soft port
(read-char p-in)
;; CHECK-NEXT: #\b

;; Test 7: Test soft port with string output procedure
(define output-buffer "")
(define p-str (make-soft-port
               (vector
                (lambda (c) (set! output-buffer (string-append output-buffer (string c))))
                (lambda (s) (set! output-buffer (string-append output-buffer s)))
                (lambda () #t)
                (lambda () #f)
                (lambda () #t))
               "w"))
(write-char #\H p-str)
(write-char #\i p-str)
output-buffer
;; CHECK-NEXT: "Hi"

;; Test 8: Test soft port with character sequence
(define char-seq '(#\1 #\2 #\3))
(define p-seq (make-soft-port
               (vector
                (lambda (c) #t)
                (lambda (s) #t)
                (lambda () #t)
                (lambda () (if (null? char-seq)
                               #f
                               (let ((ch (car char-seq)))
                                 (set! char-seq (cdr char-seq))
                                 ch)))
                (lambda () #t))
               "r"))
(read-char p-seq)
;; CHECK-NEXT: #\1
(read-char p-seq)
;; CHECK-NEXT: #\2
(read-char p-seq)
;; CHECK-NEXT: #\3

;; Test 9: Test EOF from soft port
(eof-object? (read-char p-seq))
;; CHECK-NEXT: #t

;; Test 10: Test soft port close procedure
(define close-called #f)
(define p-close (make-soft-port
                 (vector
                  (lambda (c) #t)
                  (lambda (s) #t)
                  (lambda () #t)
                  (lambda () #f)
                  (lambda () (set! close-called #t)))
                 "w"))
(close-output-port p-close)
close-called
;; CHECK-NEXT: #t

;; Test 11: Test soft port flush procedure
(define flush-called #f)
(define p-flush (make-soft-port
                 (vector
                  (lambda (c) #t)
                  (lambda (s) #t)
                  (lambda () (set! flush-called #t))
                  (lambda () #f)
                  (lambda () #t))
                 "w"))
(force-output p-flush)
flush-called
;; CHECK-NEXT: #t
