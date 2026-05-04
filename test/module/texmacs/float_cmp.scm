;; Test float comparison with negative non-integer values
(display (< -1.5 -1.0)) (newline)        ;; expected: #t
(if (not (< -1.5 -1.0)) (exit 1))
(display (> -1.0 -1.5)) (newline)        ;; expected: #t
(if (not (> -1.0 -1.5)) (exit 1))
(display (<= -1.0 -1.5)) (newline)       ;; expected: #f, bug would give #t
(if (<= -1.0 -1.5) (exit 1))
(display (>= -1.5 -1.0)) (newline)       ;; expected: #f, bug would give #t
(if (>= -1.5 -1.0) (exit 1))

;; Test mixed int/float comparisons
(display (< 3 3.5)) (newline)            ;; expected: #t
(if (not (< 3 3.5)) (exit 1))
(display (> 3.5 3)) (newline)            ;; expected: #t
(if (not (> 3.5 3)) (exit 1))
(display (< 3.5 3)) (newline)            ;; expected: #f
(if (< 3.5 3) (exit 1))
(display (> 3 3.5)) (newline)            ;; expected: #f
(if (> 3 3.5) (exit 1))

;; Chained comparisons with floats
(display (< -2.5 -1.0 0 1.5 3.0)) (newline)  ;; expected: #t
(if (not (< -2.5 -1.0 0 1.5 3.0)) (exit 1))
(display (< -1.0 -2.5 0)) (newline)          ;; expected: #f
(if (< -1.0 -2.5 0) (exit 1))

(exit 0)
