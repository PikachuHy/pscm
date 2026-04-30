;; RUN: %pscm_cc --test %s | FileCheck %s
;;
;; 测试 dynamic-wind + continuation 的 wind handler 序列。
;; 检验 scm_dynthrow 中 unwind/rewind 是否在 continuation 跳转前正确执行。
;;
;; 关键验证点:
;;   1. 正常退出时 wind handler 顺序正确 (in1→in2→in3→out3→out2→out1)
;;   2. continuation 调用时 unwind/rewind 触发正确的 handler
;;   3. continuation 调用后,执行流回到捕获点 ('stepA-done' 出现 3 次)
;;   4. continuation 多 shot 调用各自产生正确的 wind 序列

(define wind-path '())
(define (push s) (set! wind-path (cons s wind-path)))

(define saved-k #f)

;; Scenario A: 正常退出 — call/cc 第一次返回 'body-A
;; CHECK: stepA-done
(let ((r (dynamic-wind
           (lambda () (push 'in1))
           (lambda ()
             (dynamic-wind
               (lambda () (push 'in2))
               (lambda ()
                 (dynamic-wind
                   (lambda () (push 'in3))
                   (lambda ()
                     (call/cc (lambda (k) (set! saved-k k) 'body-A)))
                   (lambda () (push 'out3))))
               (lambda () (push 'out2))))
           (lambda () (push 'out1)))))
  (display "stepA-done"))

;; CHECK: wind-A= (in1 in2 in3 out3 out2 out1)
(write (list 'wind-A= (reverse wind-path)))
(newline)

;; Scenario B: 从顶层调用 continuation (第一次 multi-shot)
;; continuation 跳转使 stepA-done 再次出现,而不是 stepB-done
;; CHECK-NOT: stepB-done
;; CHECK: stepA-done
(set! wind-path '())
(let ((r (saved-k 'body-B)))
  (display "stepB-done"))

;; CHECK: wind-B= (in1 in2 in3 out3 out2 out1)
(write (list 'wind-B= (reverse wind-path)))
(newline)

;; Scenario C: 同一 continuation 再次调用 (第二次 multi-shot)
;; CHECK-NOT: stepC-done
;; CHECK: stepA-done
(set! wind-path '())
(let ((r (saved-k 'body-C)))
  (display "stepC-done"))

;; CHECK: wind-C= (in1 in2 in3 out3 out2 out1)
(write (list 'wind-C= (reverse wind-path)))
(newline)
