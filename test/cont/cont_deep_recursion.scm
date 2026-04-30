;; RUN: %pscm_cc --test %s | FileCheck %s
;;
;; 测试深层递归 + continuation 捕获与调用。
;; 50 层递归足以穿越多层 C 栈帧，此前在 ASan 下会触发 stack-buffer-underflow。

(define captured #f)

(define (recurse n)
  (if (= n 0)
      (call/cc (lambda (k) (set! captured k) 'at-bottom))
      (recurse (- n 1))))

;; CHECK: at-bottom
(recurse 50)

;; 从浅层调用 continuation（触发 grow_stack + copy_stack_and_call）
;; CHECK: 42
(captured 42)

;; 多 shot 调用
;; CHECK: 99
(captured 99)

;; 从不同递归深度调用
(define (recurse-and-call n)
  (if (= n 0)
      (captured 'from-depth)
      (recurse-and-call (- n 1))))

;; CHECK: from-depth
(recurse-and-call 30)
