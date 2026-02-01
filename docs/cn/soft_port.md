# Soft Port 实现

本文档描述 pscm_cc 中 `make-soft-port` 的实现，参考 Guile 1.8 的实现。

**状态**：✅ 已完成

## 一、设计目标

1. **自定义端口**：实现 `make-soft-port`，允许用户创建带有自定义 I/O 过程的端口
2. **灵活 I/O**：支持输入、输出和双向端口
3. **过程接口**：通过向量传递自定义 I/O 过程
4. **Guile 兼容性**：与 Guile 1.8 的行为保持一致

## 二、核心设计

### 2.1 端口类型扩展

在 `PortType` 枚举中添加了 `PORT_SOFT` 类型：

```c
enum PortType {
  PORT_FILE,
  PORT_STRING,
  PORT_SOFT  // 新增：软端口
};
```

### 2.2 端口结构扩展

在 `SCM_Port` 结构中添加了 `soft_procedures` 字段：

```c
struct SCM_Port {
  PortType port_type;
  bool is_input;
  bool is_output;
  bool is_closed;
  FILE *file;                    // 文件端口
  SCM_String *string_data;       // 字符串端口
  SCM_Vector *soft_procedures;   // 软端口：过程向量
  // ...
};
```

### 2.3 过程向量格式

软端口的过程向量包含 5 或 6 个过程：

- `[0]`：字符输出过程 `(lambda (char) ...)` - 接受一个字符
- `[1]`：字符串输出过程 `(lambda (string) ...)` - 接受一个字符串
- `[2]`：刷新过程 `(lambda () ...)` - 无参数 thunk
- `[3]`：字符读取过程 `(lambda () ...)` - 无参数 thunk，返回字符或 `#f`
- `[4]`：关闭过程 `(lambda () ...)` - 无参数 thunk
- `[5]`（可选）：字符就绪过程 `(lambda () ...)` - 无参数 thunk，返回可用字符数

### 2.4 模式字符串

模式字符串控制端口的输入/输出行为：

- `"r"`：只读端口
- `"w"`：只写端口
- `"rw"`：读写端口

## 三、API 说明

### 3.1 make-soft-port

创建软端口：

```scheme
(define p (make-soft-port
           (vector
            (lambda (c) (display c))        ; 字符输出
            (lambda (s) (display s))        ; 字符串输出
            (lambda () #t)                  ; 刷新
            (lambda () #\a)                 ; 字符读取
            (lambda () #t))                 ; 关闭
           "rw"))                           ; 模式
```

### 3.2 使用示例

#### 输出到字符串缓冲区

```scheme
(define output-buffer "")
(define p (make-soft-port
           (vector
            (lambda (c) (set! output-buffer (string-append output-buffer (string c))))
            (lambda (s) (set! output-buffer (string-append output-buffer s)))
            (lambda () #t)
            (lambda () #f)
            (lambda () #t))
           "w"))
(write-char #\H p)
(write-char #\i p)
output-buffer  ; => "Hi"
```

#### 从字符序列读取

```scheme
(define char-seq '(#\1 #\2 #\3))
(define p (make-soft-port
           (vector
            (lambda (c) #t)
            (lambda (s) #t)
            (lambda () #t)
            (lambda () (if (null? char-seq)
                          #f
                          (let ((ch (car char-seq)))
                            (set! char-seq (cdr char-seq))
                            ch)))
            (lambda () #t))
           "r"))
(read-char p)  ; => #\1
(read-char p)  ; => #\2
(read-char p)  ; => #\3
```

## 四、实现细节

### 4.1 端口操作适配

所有端口操作函数都已适配软端口：

- `read_char_from_port`：调用 `soft_procedures[3]` 读取字符
- `write_char_to_port_string`：调用 `soft_procedures[0]` 写入字符
- `flush_port`：调用 `soft_procedures[2]` 刷新
- `close-input-port` / `close-output-port`：调用 `soft_procedures[4]` 关闭

### 4.2 过程调用

使用统一的辅助函数 `call_soft_port_thunk` 调用无参数过程：

```c
static SCM *call_soft_port_thunk(SCM *proc) {
  if (!proc || is_falsy(proc)) {
    return nullptr;
  }
  
  if (is_proc(proc)) {
    SCM_Procedure *proc_obj = cast<SCM_Procedure>(proc);
    return apply_procedure(g_env.parent ? g_env.parent : &g_env, proc_obj, nullptr);
  } else if (is_func(proc)) {
    SCM_Function *func = cast<SCM_Function>(proc);
    SCM_List func_call;
    func_call.data = proc;
    func_call.next = nullptr;
    func_call.is_dotted = false;
    return eval_with_func(func, &func_call);
  }
  
  return nullptr;
}
```

### 4.3 模式处理

根据模式字符串设置 `is_input` 和 `is_output` 标志：

```c
port->is_input = (strchr(modes, 'r') != nullptr);
bool is_output_mode = (strchr(modes, 'w') != nullptr || strchr(modes, 'a') != nullptr);
port->is_output = is_output_mode;
```

## 五、测试

测试文件：`test/base/make_soft_port_tests.scm`

测试覆盖：
- 创建读写软端口
- 写入和读取操作
- 输入/输出端口模式
- 字符串缓冲区收集
- 字符序列读取
- EOF 处理
- 关闭和刷新过程调用

## 六、参考

- Guile 1.8 源码：`libguile/ports.c`
- 测试用例：`test/base/make_soft_port_tests.scm`
