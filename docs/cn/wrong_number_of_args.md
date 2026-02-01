# Wrong Number of Arguments 错误处理

本文档描述 pscm_cc 中 `wrong-number-of-args` 错误处理的实现，参考 Guile 1.8 的实现。

**状态**：✅ 已完成

## 一、设计目标

1. **参数检查**：在函数调用时检查参数数量
2. **错误抛出**：当参数数量不匹配时，抛出 `wrong-number-of-args` 错误
3. **错误捕获**：支持使用 `catch` 捕获 `wrong-number-of-args` 错误
4. **Guile 兼容性**：与 Guile 1.8 的错误格式保持一致

## 二、核心设计

### 2.1 错误格式

`wrong-number-of-args` 错误的格式与 Guile 1.8 兼容：

```scheme
(throw 'wrong-number-of-args #f "Wrong number of arguments to ~A" proc #f)
```

参数列表格式：`(caller message proc extra)`
- `caller`：调用者（通常为 `#f`）
- `message`：错误消息字符串（`"Wrong number of arguments to ~A"`）
- `proc`：被调用的过程对象
- `extra`：额外信息（通常为 `#f`）

### 2.2 错误抛出函数

```c
[[noreturn]] void throw_wrong_number_of_args(SCM_Function *func, 
                                              SCM_List *got_args, 
                                              SCM *original_call) {
  // 创建 wrong-number-of-args key
  SCM *wrong_key = wrap(make_sym("wrong-number-of-args"));
  
  // 构建错误消息和参数列表
  // ...
  
  // 抛出错误
  scm_throw(wrong_key, wrap(args_list));
}
```

### 2.3 参数检查

在 `eval_with_func` 函数中添加参数数量检查：

```c
if (func->n_args == 1) {
  // 检查参数是否缺失
  if (!l->next) {
    throw_wrong_number_of_args(func, nullptr, nullptr);
  }
  // 检查参数是否过多
  if (l->next->next) {
    throw_wrong_number_of_args(func, l->next, nullptr);
  }
  return eval_with_func_1(func, l->next->data);
}
```

## 三、API 说明

### 3.1 捕获错误

使用 `catch` 捕获 `wrong-number-of-args` 错误：

```scheme
(catch 'wrong-number-of-args
       (lambda () (car))
       (lambda (type caller message opts extra)
         type))
;; => wrong-number-of-args
```

### 3.2 检查错误消息格式

```scheme
(catch 'wrong-number-of-args
       (lambda () (car))
       (lambda (type caller message opts extra)
         message))
;; => "Wrong number of arguments to ~A"
```

### 3.3 获取过程对象

```scheme
(catch 'wrong-number-of-args
       (lambda () (car))
       (lambda (type caller message opts extra)
         (if (procedure? opts)
             'procedure
             'not-procedure)))
;; => procedure
```

### 3.4 实际应用示例

检查错误消息格式（用于兼容性检查）：

```scheme
(define old-format?
  (catch 'wrong-number-of-args
	 (lambda () (car))
	 (lambda (type caller message opts extra)
	   (let next ((l (string->list message)))
	     (cond ((null? l) #f)
		   ((char=? #\% (car l)) #t)
		   (else (next (cdr l))))))))
old-format?  ; => #f (新格式，消息中没有 % 字符)
```

## 四、实现细节

### 4.1 参数检查位置

参数检查在 `eval_with_func` 函数中进行，支持：
- 0 个参数：检查是否提供了参数
- 1 个参数：检查参数是否缺失或过多
- 2 个参数：检查参数是否缺失或过多
- 3 个参数：检查参数是否缺失或过多

### 4.2 错误信息构建

错误信息按照 Guile 1.8 的格式构建：

```c
// Build args list: (#f "Wrong number of arguments to ~A" proc #f)
SCM_List *args_list = make_list(scm_bool_false());  // caller (#f)

// Add message string
SCM *msg_str = scm_from_c_string(msg, (int)strlen(msg));
args_list->next = make_list(msg_str);

// Add procedure (the function that was called)
SCM *proc_wrapped = func ? wrap(func) : scm_bool_false();
args_list->next->next = make_list(proc_wrapped);

// Add extra (#f)
args_list->next->next->next = make_list(scm_bool_false());
```

### 4.3 与 catch 集成

`wrong-number-of-args` 错误可以通过 `catch` 机制捕获，与其他错误类型一样。

## 五、测试

测试文件：`test/base/wrong_number_of_args_tests.scm`

测试覆盖：
- 捕获无参数调用错误
- 捕获参数过多错误
- 检查错误消息格式
- 检查错误参数（caller、message、proc、extra）
- 检查 old-format? 函数
- 正常执行不触发错误
- 使用 #t 标签捕获所有错误
- 不同函数的参数错误

## 六、使用场景

### 6.1 参数验证

在函数调用时自动检查参数数量，提供清晰的错误信息。

### 6.2 兼容性检查

可以检查错误消息格式，用于判断是否使用旧格式的错误消息。

### 6.3 错误恢复

可以通过 `catch` 捕获错误并进行恢复处理。

## 七、参考

- Guile 1.8 源码：`libguile/eval.c`
- 测试用例：`test/base/wrong_number_of_args_tests.scm`
- 相关功能：catch/throw 异常处理机制
