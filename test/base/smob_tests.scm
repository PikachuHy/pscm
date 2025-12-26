;; RUN: %pscm_cc --test %s | FileCheck %s
;; Test smob (Small Object) functionality

;; Test 1: Create smob type and instance
(define image-tag (scm-make-smob-type "image" 0))
(define img (scm-make-smob image-tag 42))

;; Test 2: Check if it's a smob
;; CHECK: #t
(scm-smob? img)

;; CHECK-NEXT: #f
(scm-smob? 42)

;; Test 3: Access smob data
;; CHECK-NEXT: 42
(scm-smob-data img)

;; Test 4: Print smob
;; CHECK-NEXT: #<image 42>
(write img)
(newline)

;; Test 5: Create another smob type
(define point-tag (scm-make-smob-type "point" 0))
(define p (scm-make-smob point-tag 100))
;; CHECK-NEXT: #<point 100>
(write p)
(newline)

;; Test 6: Test eq? and equal? with smobs
(define img2 (scm-make-smob image-tag 42))
;; CHECK-NEXT: #f
(eq? img img2)

;; CHECK-NEXT: #f
(equal? img img2)

;; CHECK-NEXT: #t
(eq? img img)

