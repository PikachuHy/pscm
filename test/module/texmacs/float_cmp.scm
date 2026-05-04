;; Test float comparison with negative non-integer values
(display (< -1.5 -1.0)) (newline)        ;; expected: #t
(display (> -1.0 -1.5)) (newline)        ;; expected: #t
(display (<= -1.5 -1.0)) (newline)       ;; expected: #t
(display (>= -1.0 -1.5)) (newline)       ;; expected: #t

;; Test mixed int/float comparisons
(display (< 3 3.5)) (newline)            ;; expected: #t
(display (> 3.5 3)) (newline)            ;; expected: #t
(display (< 3.5 3)) (newline)            ;; expected: #f
(display (> 3 3.5)) (newline)            ;; expected: #f

;; Chained comparisons with floats
(display (< -2.5 -1.0 0 1.5 3.0)) (newline)  ;; expected: #t
(display (< -1.0 -2.5 0)) (newline)          ;; expected: #f
