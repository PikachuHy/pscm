;; Test: file port basics
;; Use read-char + read cycle instead of read-line (which may not exist)
(define tmpfile "test_port_tmp.txt")
(define out (open-output-file tmpfile))
(display "test content" out)
(newline out)
(close-output-port out)

(define in (open-input-file tmpfile))
;; Read first line using read (for the symbol) and read-char for newline
(display (read in)) (newline)
(read-char in)  ;; consume newline
(close-input-port in)
(display "FILEPORT-PASS") (newline)
