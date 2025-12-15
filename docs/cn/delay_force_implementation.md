## 延迟求值实现方案：`delay` / `force`（Promise）

本文档描述在 `pscm cc` 中实现 Scheme 标准的延迟求值机制：`delay` 和 `force`。目标是：

- 提供与 R4RS/R5RS 一致的语义；
- 设计一个简单的 Promise 表示，尽量贴近 Guile 1.8 的设计思想，但不引入线程/互斥锁等复杂度；
- 兼容当前的类型系统和求值器结构。

### 1. 需求与语义

- **接口**
  - 特殊形式：`(delay <expr>)` —— 返回一个 Promise，对 `<expr>` 的求值被延迟；
  - 函数：`(force promise)` —— 若 Promise 尚未求值，则在首次调用时求值并缓存结果；之后再 `force` 同一个 Promise 时直接返回缓存值，不重复求值。
- **R4RS/R5RS 测试行为**
  - `(force (delay (+ 1 2)))  => 3`
  - 多次 `force` 同一 Promise，表达式只执行一次（可通过副作用测试）；
  - 支持用 Promise 构建懒序列（stream）：
    - 例如 `a-stream` 形如 `(cons n (delay (next (+ n 1))))`，通过 `force` 展开 `cdr`。

### 2. 与 Guile 1.8 的对齐思路

Guile 1.8 的设计要点（参考 `guile/libguile/eval.c`）：

- 语法层面：`delay` 是一个宏/语法，实质将 `(delay <expr>)` 转换为一个“无参闭包”（thunk）的 Promise 表示；
- 运行时：Promise 内部保存：
  - 一个 thunk（闭包）；
  - 一个“已计算”标记；
  - 以及一个互斥锁用于多线程场景；
- `force` 的逻辑：
  1. 检查是否已计算；
  2. 如未计算，则调用内部 thunk；
  3. 将结果写回 Promise，并标记为已计算；
  4. 返回结果。

在本项目中我们不需要线程相关的互斥锁，实现可以更简化。

### 3. 设计方案（pscm cc）

#### 3.1 Promise 类型表示

- 在 `SCM::Type` 中新增 `PROMISE` 枚举值；
- 新增结构体：

  ```c++
  struct SCM_Promise {
    SCM *thunk;      // 一个无参过程（PROC 或 FUNC），表示延迟计算
    SCM *value;      // 计算完成后缓存的值；未计算时为 nullptr
    bool is_forced;  // 是否已经执行过计算
  };
  ```

- 一个 Promise 在 `SCM` 层的表示：
  - `type = SCM::PROMISE`
  - `value = SCM_Promise*`

#### 3.2 delay 的实现策略

- 为了简单且与现有架构对齐，`delay` 直接在 C++ 中作为**特殊形式**实现：
  - 语法：`(delay <expr>)`
  - 实现文件：`eval.cc` 或单独 `delay.cc`（初版可放在 `eval.cc` 按特殊形式风格实现）；
  - 实现逻辑：
    1. 收到形如 `(delay expr)` 的 AST 列表；
    2. 构造一个无参过程（thunk），其 body 即为 `<expr>`：
       - 可以复用已有的过程构造逻辑（类似 `lambda`），或者简单构造一个 `SCM_Procedure`，参数列表为空，body 只有一个 `<expr>`；
    3. 用该 thunk 创建一个 `SCM_Promise`：
       - `thunk` 指向封装好的无参过程（或 `SCM_Function`）；
       - `value = nullptr`，`is_forced = false`；
    4. 返回一个 `SCM`，类型为 `PROMISE`。

- 这样 `delay` 本质上就是“把 `<expr>` 封装成无参过程 + Promise 容器”。

#### 3.3 force 的实现策略

- 在 C++ 中实现内置函数：
  - 签名：`SCM *scm_c_force(SCM *promise);`
  - 注册：在新的 `init_delay_force()`（或直接 `init_eval`）中 `scm_define_function("force", 1, 0, 0, scm_c_force);`
- 实现逻辑：
  1. 检查参数类型：
     - 若不是 `PROMISE`，调用 `type_error(promise, "promise");`（或 `eval_error("force: expected promise")`）；
  2. 若 `is_forced == true`，直接返回 `value`；
  3. 否则：
     - 从 Promise 中取出 `thunk`；
     - 根据 `thunk` 的类型调用：
       - 如果是 `PROC`：使用 `apply_procedure` 在当前环境（或顶层环境）下以空参数列表调用；
       - 如果是 `FUNC`：构造形如 `(<thunk>)` 的调用列表，用 `eval_with_func` 执行；
     - 将返回值写入 `promise->value`，并设置 `is_forced = true`；
     - 返回该值。

- 这样可以保证：
  - Promise 只在第一次 `force` 时真正执行内部 thunk；
  - 后续多次 `force` 都直接返回缓存结果；
  - R4RS 测试里那种自递归 `force` 的写法也能按预期工作。

#### 3.4 初始化与接口暴露

- 新增初始化函数，例如：
  - `void init_delay_force();`
  - 在 `pscm.h` 中声明；
  - 在 `init_scm()` 里在合适位置（如 `init_eval` 前后）调用。
- 在 `init_delay_force()` 中：
  - 注册 `force`；
  - 如有需要，为 `promise?` 等扩展留出口（当前 r4rs/r5rs 测试只用到 `delay`/`force`）。

### 4. 测试与兼容性

- 使用现有测试命令：

  ```bash
  ./pscm_cc --test test/r4rs/r4rstest.scm
  ./pscm_cc --test test/r5rs/r5rstest.scm
  ```

- 重点验证：
  - `SECTION 6 9` 中的所有 `delay`/`force` 相关测试全部通过；
  - 特别是：
    - 多次 `force` 同一 Promise 只计算一次；
    - 使用 Promise 构造的流在递归场景下行为正确。

### 5. 与后续扩展的关系

- 如果未来引入多线程或更复杂的调度机制，可以参考 Guile 1.8 的设计，在 `SCM_Promise` 中加入锁或其他同步原语；
- 也可以在此基础上添加：
  - `promise?` 谓词；
  - 更丰富的 stream 库。


