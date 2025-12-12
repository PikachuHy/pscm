# pscm dynamic-wind 实现方案

## 概述

`dynamic-wind` 是 Scheme 中用于管理动态上下文（dynamic context）的特殊形式。它确保在进入和退出某个动态作用域时，特定的清理和初始化代码能够正确执行，即使在非局部控制流转移（如 continuation 调用）的情况下也是如此。

### 为什么需要 dynamic-wind？

在 Scheme 中，continuation 允许程序"跳回"到之前的执行点。但是，如果在执行过程中有资源需要清理（如打开的文件、设置的全局变量等），简单的 continuation 可能会导致资源泄漏或状态不一致。

`dynamic-wind` 提供了三个过程：
- `in_guard`: 进入动态作用域时调用
- `thunk`: 实际执行的代码
- `out_guard`: 退出动态作用域时调用

无论控制流如何转移（正常返回、异常、continuation 调用），`in_guard` 和 `out_guard` 都会被正确调用。

## 设计原理

### Wind Chain（风链）

实现 `dynamic-wind` 的核心是维护一个 **wind chain**（风链），它是一个链表，记录了所有活跃的 `dynamic-wind` 调用。每个节点存储一个 `(in_guard . out_guard)` 对。

```
wind chain: [entry1] -> [entry2] -> [entry3] -> nullptr
             |            |            |
          (in1.out1)  (in2.out2)  (in3.out3)
```

### 基本执行流程

1. **正常执行**：
   ```
   dynamic-wind in_guard thunk out_guard
   ```
   - 调用 `in_guard`
   - 将 `(in_guard . out_guard)` 添加到 wind chain
   - 执行 `thunk`
   - 从 wind chain 移除该条目
   - 调用 `out_guard`

2. **Continuation 调用**：
   当调用一个 continuation 时，需要：
   - **Unwind**：从当前 wind chain 回退到 continuation 创建时的 wind chain（调用相应的 `out_guard`）
   - **Rewind**：从公共前缀恢复到 continuation 的 wind chain（调用相应的 `in_guard`）

## 数据结构

### 全局 Wind Chain

```c
extern SCM_List *g_wind_chain;  // 全局 wind chain
```

`g_wind_chain` 是一个链表，每个节点存储一个 `(in_guard . out_guard)` 对。

### Continuation 结构扩展

```c
struct SCM_Continuation {
  jmp_buf cont_jump_buffer;
  size_t stack_len;
  void *stack_data;
  void *dst;
  SCM *arg;
  SCM_List *wind_chain;  // 保存创建时的 wind chain
};
```

在创建 continuation 时，会保存当前的 wind chain，以便在调用 continuation 时能够正确恢复。

## 关键函数

### 1. `eval_dynamic_wind` - 处理 dynamic-wind 特殊形式

```c
SCM *eval_dynamic_wind(SCM_Environment *env, SCM_List *l)
```

**功能**：执行 `dynamic-wind` 特殊形式

**流程**：
1. 求值三个参数：`in_guard`、`thunk`、`out_guard`
2. 验证它们都是过程
3. 调用 `in_guard`
4. 创建 wind entry：`(in_guard . out_guard)`，并添加到 wind chain
5. 执行 `thunk`
6. 恢复旧的 wind chain
7. 调用 `out_guard`
8. 返回 `thunk` 的结果

### 2. `find_common_prefix` - 查找公共前缀

```c
SCM_List *find_common_prefix(SCM_List *chain1, SCM_List *chain2)
```

**功能**：找到两个 wind chain 的公共前缀

**算法**：
- 从两个链表的头部开始比较
- 如果节点的 `data` 指针相同，继续比较下一个节点
- 返回公共部分的副本

**用途**：在调用 continuation 时，找到当前 wind chain 和目标 wind chain 的公共部分，以便确定需要 unwind 和 rewind 的部分。

### 3. `unwind_wind_chain` - 回退 Wind Chain

```c
SCM_List *unwind_wind_chain(SCM_List *target)
```

**功能**：从当前 wind chain 回退到目标 wind chain

**流程**：
1. 找到当前 wind chain 和目标 wind chain 的公共前缀
2. 收集需要 unwind 的条目（不在公共前缀中的条目）
3. 按正确顺序（从后往前）调用每个条目的 `out_guard`
4. 更新 `g_wind_chain` 为公共前缀
5. 返回公共前缀

**关键点**：需要按 LIFO（后进先出）顺序调用 `out_guard`，因为 wind chain 是栈式结构。

### 4. `rewind_wind_chain` - 恢复 Wind Chain

```c
void rewind_wind_chain(SCM_List *common, SCM_List *target)
```

**功能**：从公共前缀恢复到目标 wind chain

**流程**：
1. 找到目标 wind chain 中不在公共前缀中的条目
2. 按正确顺序（从前往后）调用每个条目的 `in_guard`
3. 将每个条目添加到 `g_wind_chain`

**关键点**：需要按 FIFO（先进先出）顺序调用 `in_guard`，以保持正确的嵌套顺序。

### 5. `copy_wind_chain` - 复制 Wind Chain

```c
SCM_List *copy_wind_chain(SCM_List *chain)
```

**功能**：深拷贝一个 wind chain

**用途**：在创建 continuation 时保存当前的 wind chain，避免后续修改影响已保存的 continuation。

## 执行流程详解

### 场景 1：正常执行

```scheme
(dynamic-wind
  (lambda () (add 'connect))
  (lambda () (add 'talk))
  (lambda () (add 'disconnect)))
```

**执行步骤**：
1. 调用 `in_guard` → `(add 'connect)` → path = `(connect)`
2. 添加 wind entry 到 wind chain
3. 执行 `thunk` → `(add 'talk)` → path = `(talk connect)`
4. 从 wind chain 移除 wind entry
5. 调用 `out_guard` → `(add 'disconnect)` → path = `(disconnect talk connect)`

### 场景 2：Continuation 调用

```scheme
(let ((path '())
      (c #f))
  (let ((add (lambda (s) (set! path (cons s path)))))
    (dynamic-wind
      (lambda () (add 'connect))
      (lambda ()
        (add (call-with-current-continuation
              (lambda (c0)
                (set! c c0)
                'talk1))))
      (lambda () (add 'disconnect)))
    (if (< (length path) 4)
        (c 'talk2)
        (reverse path))))
```

**执行步骤**：

**第一次执行**：
1. 调用 `in_guard` → path = `(connect)`
2. 添加 wind entry 到 wind chain
3. 执行 `thunk`：
   - 调用 `call/cc`，保存 continuation（包含当前的 wind chain）
   - `add 'talk1` → path = `(talk1 connect)`
4. 从 wind chain 移除 wind entry
5. 调用 `out_guard` → path = `(disconnect talk1 connect)`
6. 检查 `length path < 4` → 3 < 4，调用 continuation

**调用 continuation**：
1. **Unwind**：
   - 当前 wind chain = `nullptr`（已退出 dynamic-wind）
   - 目标 wind chain = continuation 保存的 wind chain（包含 dynamic-wind）
   - 公共前缀 = `nullptr`
   - 无需调用 `out_guard`（当前 wind chain 为空）
2. **Rewind**：
   - 从 `nullptr` 恢复到包含 dynamic-wind 的 wind chain
   - 调用 `in_guard` → path = `(connect disconnect talk1 connect)`
   - 添加 wind entry 到 wind chain
3. 继续执行，`add 'talk2` → path = `(talk2 connect disconnect talk1 connect)`
4. 再次退出 dynamic-wind：
   - 从 wind chain 移除 wind entry
   - 调用 `out_guard` → path = `(disconnect talk2 connect disconnect talk1 connect)`
5. 检查 `length path < 4` → 5 < 4 为假，执行 `reverse path`
6. 最终结果：`(connect talk1 disconnect connect talk2 disconnect)`

## 与 Continuation 的集成

### Continuation 创建时

在 `scm_make_continuation` 中：

```c
// Save current wind chain
cont->wind_chain = copy_wind_chain(g_wind_chain);
```

保存当前的 wind chain，以便后续恢复。

### Continuation 调用时

在 `scm_dynthrow` 中：

```c
// Handle wind chain: unwind current to common prefix, then rewind to continuation's wind chain
SCM_List *common = unwind_wind_chain(continuation->wind_chain);
rewind_wind_chain(common, continuation->wind_chain);
```

1. 先 unwind：回退到公共前缀
2. 再 rewind：恢复到 continuation 的 wind chain

这确保了在跳转到 continuation 时，所有相关的 `in_guard` 和 `out_guard` 都被正确调用。

## 实现细节

### Wind Entry 的存储格式

每个 wind entry 是一个 pair：
```
(in_guard . out_guard)
```

存储在 wind chain 的节点中：
```c
SCM_List *wind_entry = make_list(in_guard_val);
wind_entry->next = make_list(out_guard_val);
```

### 公共前缀的查找

通过比较节点的 `data` 指针来查找公共前缀：
```c
while (c1 && c2 && c1->data == c2->data) {
    common_count++;
    c1 = c1->next;
    c2 = c2->next;
}
```

如果两个节点的 `data` 指针相同，说明它们指向同一个 `(in_guard . out_guard)` 对，是公共部分。

### Unwind 的顺序

Unwind 时需要按 LIFO 顺序调用 `out_guard`：

```c
// 收集需要 unwind 的条目（反向收集）
while (c) {
    if (!in_common) {
        auto new_entry = make_list(c->data);
        new_entry->next = to_unwind;
        to_unwind = new_entry;  // 反向链接
    }
    c = c->next;
}

// 按反向顺序调用（即正确的 LIFO 顺序）
while (to_unwind) {
    // 调用 out_guard
    to_unwind = to_unwind->next;
}
```

### Rewind 的顺序

Rewind 时需要按 FIFO 顺序调用 `in_guard`：

```c
// 收集需要 rewind 的条目（反向收集）
while (c) {
    auto new_entry = make_list(c->data);
    new_entry->next = to_rewind;
    to_rewind = new_entry;  // 反向链接
    c = c->next;
}

// 按反向顺序调用（即正确的 FIFO 顺序）
while (to_rewind) {
    // 调用 in_guard
    // 添加到 wind chain
    to_rewind = to_rewind->next;
}
```

## 测试用例

### 测试 1：基本功能

```scheme
(let* ((path '())
       (add (lambda (s) (set! path (cons s path)))))
  (dynamic-wind 
    (lambda () (add 'a)) 
    (lambda () (add 'b)) 
    (lambda () (add 'c)))
  (reverse path))
```

**预期输出**：`(a b c)`

### 测试 2：与 Continuation 交互

```scheme
(let ((path '())
      (c #f))
  (let ((add (lambda (s) (set! path (cons s path)))))
    (dynamic-wind
      (lambda () (add 'connect))
      (lambda ()
        (add (call-with-current-continuation
              (lambda (c0)
                (set! c c0)
                'talk1))))
      (lambda () (add 'disconnect)))
    (if (< (length path) 4)
        (c 'talk2)
        (reverse path))))
```

**预期输出**：`(connect talk1 disconnect connect talk2 disconnect)`

这个测试验证了：
- 第一次进入：`connect` → `talk1` → `disconnect`
- 调用 continuation：`connect`（rewind）→ `talk2` → `disconnect`

## 参考实现

本实现参考了 Guile 1.8 的 `dynamic-wind` 实现，特别是：
- `libguile/dynwind.c` 中的 `scm_dynamic_wind` 函数
- `scm_i_dowinds` 函数中的 unwind/rewind 逻辑

## 总结

`dynamic-wind` 的实现通过维护一个 wind chain 来跟踪所有活跃的动态上下文。当调用 continuation 时，通过 unwind 和 rewind 操作确保所有相关的清理和初始化代码都被正确执行，从而保证了资源管理和状态一致性。

关键点：
1. **Wind Chain**：链表结构，存储所有活跃的 `(in_guard . out_guard)` 对
2. **公共前缀**：找到两个 wind chain 的公共部分，确定需要 unwind/rewind 的范围
3. **Unwind**：按 LIFO 顺序调用 `out_guard`
4. **Rewind**：按 FIFO 顺序调用 `in_guard`
5. **Continuation 集成**：在创建和调用 continuation 时正确处理 wind chain
