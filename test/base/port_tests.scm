;; RUN: cd $(dirname %pscm_cc) && %pscm_cc --test %s | FileCheck %s
(define filename "compile_commands.json")
;; Test file ports
;; CHECK: #<input: file port>
(open-input-file filename)

;; Test string input ports
;; CHECK: #\h
(let ((port (open-input-string "hello")))
  (read-char port))

;; CHECK: #\e
(let ((port (open-input-string "hello")))
  (read-char port)
  (read-char port))

;; CHECK: #t
(eof-object? (read-char (open-input-string "")))

;; CHECK: #f
(eof-object? (read-char (open-input-string "a")))

;; Test peek-char
;; CHECK: #\h
(let ((port (open-input-string "hello")))
  (peek-char port))

;; CHECK: #\h
(let ((port (open-input-string "hello")))
  (peek-char port)
  (peek-char port))

;; CHECK: #t
(let ((port (open-input-string "hello")))
  (char-ready? port))

;; CHECK: #f
(let ((port (open-input-string "")))
  (char-ready? port))

;; Test string output ports
;; CHECK: "hello"
(let ((port (open-output-string)))
  (display "hello" port)
  (get-output-string port))

;; CHECK: "123"
(let ((port (open-output-string)))
  (display 123 port)
  (get-output-string port))

;; Test call-with-input-string
;; CHECK: #\h
(call-with-input-string "hello" (lambda (port) (read-char port)))

;; CHECK: "hello"
(call-with-output-string (lambda (port) (display "hello" port)))

;; CHECK: "test123"
(call-with-output-string (lambda (port) (display "test" port) (display 123 port)))

;; Test read from string port
;; CHECK: hello
(let ((port (open-input-string "hello")))
  (read port))

;; CHECK: 123
(let ((port (open-input-string "123")))
  (read port))

;; CHECK: (a b c)
(let ((port (open-input-string "(a b c)")))
  (read port))

;; Test close ports
;; CHECK: #<input: file port>
(let ((port (open-input-file filename)))
  (display port)
  (close-input-port port))

;; CHECK: #<output: string port>
(let ((port (open-output-string)))
  (display port)
  (close-output-port port))

;; Test call-with-input-file
;; CHECK: #<input: file port>
(call-with-input-file filename (lambda (port) port))

;; Test call-with-output-file
;; CHECK: #<output: file port>
(call-with-output-file
  "/tmp/test_output.txt"
  (lambda (port)
    (display port)
    (display "test" port)))
