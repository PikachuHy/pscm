;; Test: string port round-trip
(define s (open-output-string))
(display "hello" s)
(display " world" s)
(define result (get-output-string s))
(display result) (newline)
(display (string=? result "hello world")) (newline)

;; Test: input string port
(define in (open-input-string "scheme data"))
(display (read in)) (newline)
;; Read the space character
(display (read-char in)) (newline)
(display "STRINGPORT-PASS") (newline)
