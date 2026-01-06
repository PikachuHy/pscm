;; RUN: %pscm_cc --test %s | FileCheck %s
;; Test scm_print_state functionality

;; Test 1: Create smob type and test printing with print_state
(define image-tag (scm-make-smob-type "image" 0))
(define img (scm-make-smob image-tag 42))

;; Test write mode (writingp = true)
;; CHECK: #<image 42>
(write img)
(newline)

;; Test 2: Create smob with size > 0 (pointer-based)
;; Note: scm-make-smob with size > 0 requires a pointer, but we're using
;; scm-make-smob which takes a number, so it will store 0 as the pointer
(define box-tag (scm-make-smob-type "box" 8))
(define box-data (scm-make-smob box-tag 0))
;; CHECK-NEXT: #<box 0x0>
(write box-data)
(newline)

;; Test 3: Multiple smob types with different data
(define point-tag (scm-make-smob-type "point" 0))
(define p1 (scm-make-smob point-tag 10))
(define p2 (scm-make-smob point-tag 20))

;; CHECK-NEXT: #<point 10>
(write p1)
(newline)

;; CHECK-NEXT: #<point 20>
(write p2)
(newline)

;; Test 4: Test that print_state is properly initialized
;; The print_state should handle write mode correctly
;; CHECK-NEXT: #<image 42>
(write img)
(newline)

;; Test 5: Test with different smob instances
(define counter-tag (scm-make-smob-type "counter" 0))
(define c1 (scm-make-smob counter-tag 1))
(define c2 (scm-make-smob counter-tag 2))
(define c3 (scm-make-smob counter-tag 3))

;; CHECK-NEXT: #<counter 1>
(write c1)
(newline)

;; CHECK-NEXT: #<counter 2>
(write c2)
(newline)

;; CHECK-NEXT: #<counter 3>
(write c3)
(newline)

;; Test 6: Verify print_state works with nested structures
;; (smob in a list)
(define my-list (list img p1 c1))
;; CHECK-NEXT: (#<image 42> #<point 10> #<counter 1>)
(write my-list)
(newline)

;; Test 7: Test print_state with vector containing smobs
(define my-vector (vector img p1 c1))
;; CHECK-NEXT: #(#<image 42> #<point 10> #<counter 1>)
(write my-vector)
(newline)

;; Test 8: Verify that print_state maintains state correctly
;; across multiple print operations
(define tag1 (scm-make-smob-type "type1" 0))
(define tag2 (scm-make-smob-type "type2" 0))
(define tag3 (scm-make-smob-type "type3" 0))

(define obj1 (scm-make-smob tag1 100))
(define obj2 (scm-make-smob tag2 200))
(define obj3 (scm-make-smob tag3 300))

;; CHECK-NEXT: #<type1 100>
(write obj1)
(newline)

;; CHECK-NEXT: #<type2 200>
(write obj2)
(newline)

;; CHECK-NEXT: #<type3 300>
(write obj3)
(newline)

;; Test 9: Test print_state with smob containing zero value
(define zero-tag (scm-make-smob-type "zero" 0))
(define zero-obj (scm-make-smob zero-tag 0))
;; CHECK-NEXT: #<zero 0>
(write zero-obj)
(newline)

;; Test 10: Test print_state with smob containing negative value
(define neg-tag (scm-make-smob-type "negative" 0))
(define neg-obj (scm-make-smob neg-tag -42))
;; CHECK-NEXT: #<negative -42>
(write neg-obj)
(newline)

