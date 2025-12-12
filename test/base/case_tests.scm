;; RUN: %pscm_cc --test %s | FileCheck %s

;; Basic case expression with number matching
;; CHECK: composite
(case (* 2 3)
  ((2 3 5 7) 'prime)
  ((1 4 6 8 9) 'composite))

;; Case with symbol matching
;; CHECK: consonant
(case (car '(c d))
  ((a e i o u) 'vowel)
  ((w y) 'semivowel)
  (else 'consonant))

;; Case with else clause (no match)
;; CHECK: other
(case 10
  ((1 2 3) 'small)
  ((4 5 6) 'medium)
  (else 'other))

;; Case with single datum in clause
;; CHECK: one
(case 1
  ((1) 'one)
  ((2) 'two)
  (else 'other))

;; Case with empty datum list (should not match)
;; CHECK: other
(case 1
  (() 'empty)
  (else 'other))

;; Case with character matching
;; CHECK: vowel
(case #\a
  ((#\a #\e #\i #\o #\u) 'vowel)
  (else 'letter))

;; Case with character not matching vowel
;; CHECK: letter
(case #\b
  ((#\a #\e #\i #\o #\u) 'vowel)
  (else 'letter))

;; Case with multiple expressions in clause
;; CHECK: 3
(case 2
  ((1) 1)
  ((2) 2 3)
  (else 0))

;; Case with no matching clause and no else (should return #f)
;; CHECK: #f
(case 99
  ((1 2 3) 'small)
  ((4 5 6) 'medium))

;; Case with string matching
;; CHECK: found
(case "test"
  (("test" "hello") 'found)
  (("world") 'not-found)
  (else 'other))

;; Case with nested expressions
;; CHECK: result
(case (car (list 'x))
  ((x y z) 'result)
  (else 'no-match))

;; Case with quoted symbols
;; CHECK: symbol-match
(case 'foo
  ((foo bar) 'symbol-match)
  ((baz qux) 'other-symbol)
  (else 'no-match))
