# pscm_cc Continuation 实现分析

## 概述

本文档分析 pscm_cc 中 continuation 的实现方式，对比 Guile 1.8 的实现，并讨论其优劣以及改进方向。

## pscm_cc 的实现

### 核心数据结构

```c
struct SCM_Continuation {
  jmp_buf cont_jump_buffer;      // setjmp/longjmp 跳转缓冲区
  size_t stack_len;               // 保存的栈大小
  void *stack_data;               // 保存的栈数据
  void *dst;                       // 目标栈位置（未使用）
  SCM *arg;                        // continuation 的参数
  SCM_List *wind_chain;            // 保存的 dynamic-wind 链
};
```

### 实现机制

#### 1. 创建 Continuation (`scm_make_continuation`)

```c
SCM *scm_make_continuation(int *first) {
  // 1. 计算当前栈大小
  long stack_size = scm_stack_size(cont_base);
  
  // 2. 分配 continuation 对象和栈数据空间
  auto cont = make_cont(stack_size, (long *)malloc(sizeof(long) * stack_size));
  
  // 3. 复制栈数据
  src = cont_base;
  src -= stack_size;
  memcpy((void *)cont->stack_data, src, sizeof(long) * cont->stack_len);
  
  // 4. 保存当前的 wind chain
  cont->wind_chain = copy_wind_chain(g_wind_chain);
  
  // 5. 使用 setjmp 保存执行上下文
  *first = !setjmp(cont->cont_jump_buffer);
  if (*first) {
    return data;  // 第一次返回，返回 continuation 对象
  }
  else {
    return cont->arg;  // 后续返回，返回传入的参数
  }
}
```

**关键点：**
- 使用 `scm_stack_size` 计算从 `cont_base` 到当前栈顶的大小
- 通过 `memcpy` 复制整个栈段
- 使用 `setjmp` 保存寄存器状态和跳转点
- 保存当前的 `wind_chain` 用于 `dynamic-wind` 支持

#### 2. 调用 Continuation (`scm_dynthrow`)

```c
void scm_dynthrow(SCM *cont, SCM *args) {
  auto continuation = cast<SCM_Continuation>(cont);
  
  // 1. 处理 wind chain：unwind 到公共前缀，然后 rewind 到 continuation 的 wind chain
  SCM_List *common = unwind_wind_chain(continuation->wind_chain);
  rewind_wind_chain(common, continuation->wind_chain);
  
  // 2. 计算目标栈位置
  long *dst = cont_base;
  long stack_top_element;
  dst -= continuation->stack_len;
  
  // 3. 检查栈空间是否足够，不够则递归增长
  if (dst <= &stack_top_element) {
    grow_stack(cont, args);
  }
  
  // 4. 复制栈并跳转
  copy_stack_and_call(continuation, args, dst);
}
```

**关键点：**
- 先处理 `dynamic-wind`：unwind 当前 wind chain 到公共前缀，然后 rewind 到 continuation 保存的 wind chain
- 检查栈空间是否足够，通过递归调用 `grow_stack` 来增长栈
- 使用 `memcpy` 复制保存的栈数据到目标位置
- 使用 `longjmp` 跳转回保存的执行点

#### 3. 栈增长机制 (`grow_stack`)

```c
void grow_stack(SCM *cont, SCM *args) {
  long growth[100];  // 在栈上分配 100 个 long 的空间
  scm_dynthrow(cont, args);  // 递归调用，重新检查栈空间
}
```

**关键点：**
- 通过在栈上分配局部数组来"增长"栈
- 递归调用 `scm_dynthrow` 重新检查栈空间
- 这是一个简单但有效的栈增长策略

### 优势

1. **实现简单**：代码量少，逻辑清晰，易于理解和维护
2. **功能完整**：支持基本的 continuation 功能，包括 `call/cc` 和 `dynamic-wind`
3. **栈管理直接**：通过直接复制栈数据实现，不需要复杂的栈管理机制
4. **内存效率**：只保存必要的栈数据，没有额外的元数据开销

### 劣势

1. **栈大小计算不精确**：使用 `long` 类型存储栈数据，可能在某些架构上不够精确
2. **缺少调试支持**：没有保存调试框架信息，无法进行栈回溯
3. **缺少 continuation barrier**：没有实现 `with-continuation-barrier`，无法限制 continuation 的作用域
4. **栈增长策略简单**：固定增长 100 个 `long`，可能不够灵活
5. **缺少架构特定优化**：没有针对特定架构（如 ia64）的特殊处理
6. **指针调整缺失**：没有处理栈中指针的调整（offset），可能导致指针失效
7. **缺少错误处理**：没有检查 continuation 是否在 critical section 中被调用

## Guile 1.8 的实现

### 核心数据结构

```c
typedef struct {
  SCM throw_value;              // continuation 的返回值
  scm_i_jmp_buf jmpbuf;         // 跳转缓冲区（可能是 setjmp 或 setcontext）
  SCM dynenv;                    // dynamic environment（wind chain）
  void *backing_store;           // ia64 架构的寄存器备份存储
  unsigned long backing_store_size;
  size_t num_stack_items;       // 保存的栈项数量
  SCM root;                      // continuation root 标识符（用于 barrier）
  scm_t_ptrdiff offset;          // 栈偏移量，用于调整栈内指针
  struct scm_t_debug_frame *dframe;  // 调试框架
  SCM_STACKITEM stack[1];        // 保存的栈数据（变长数组）
} scm_t_contregs;
```

### 实现机制对比

#### 1. 创建 Continuation

**Guile 1.8 的实现：**

```c
SCM scm_make_continuation (int *first)
{
  // 1. 计算栈大小
  stack_size = scm_stack_size (thread->continuation_base);
  
  // 2. 使用 GC 分配内存
  continuation = scm_gc_malloc (sizeof (scm_t_contregs)
                                + (stack_size - 1) * sizeof (SCM_STACKITEM),
                                "continuation");
  
  // 3. 保存各种状态
  continuation->dynenv = scm_i_dynwinds ();
  continuation->root = thread->continuation_root;
  continuation->dframe = scm_i_last_debug_frame ();
  
  // 4. 计算并保存 offset
  continuation->offset = continuation->stack - src;
  
  // 5. 复制栈数据
  memcpy (continuation->stack, src, sizeof (SCM_STACKITEM) * stack_size);
  
  // 6. 处理 ia64 架构的特殊情况
  #ifdef __ia64__
    // 保存寄存器备份存储
  #endif
  
  // 7. setjmp
  *first = !SCM_I_SETJMP (continuation->jmpbuf);
  // ...
}
```

**关键差异：**
- 使用 `SCM_STACKITEM` 而不是 `long`，更精确
- 保存 `offset` 用于调整栈内指针
- 保存调试框架信息 `dframe`
- 保存 `continuation_root` 用于 barrier 检查
- 针对 ia64 架构有特殊处理（寄存器备份存储）
- 使用 GC 管理内存

#### 2. 调用 Continuation

**Guile 1.8 的实现：**

```c
static void scm_dynthrow (SCM cont, SCM val)
{
  // 1. 检查是否在 critical section
  if (thread->critical_section_level) {
    fprintf (stderr, "continuation invoked from within critical section.\n");
    abort ();
  }
  
  // 2. 计算目标位置并检查栈空间
  dst -= continuation->num_stack_items;
  if (dst <= &stack_top_element)
    grow_stack (cont, val);
  
  // 3. 刷新寄存器窗口
  SCM_FLUSH_REGISTER_WINDOWS;
  
  // 4. 复制栈并调用
  copy_stack_and_call (continuation, val, dst);
}

static void copy_stack_and_call (scm_t_contregs *continuation, SCM val,
                                 SCM_STACKITEM * dst)
{
  // 1. 计算 wind chain 的差异
  delta = scm_ilength (scm_i_dynwinds ()) - scm_ilength (continuation->dynenv);
  
  // 2. 在 wind 过程中复制栈（使用回调函数）
  scm_i_dowinds (continuation->dynenv, delta, copy_stack, &data);
  
  // 3. 恢复调试框架
  scm_i_set_last_debug_frame (continuation->dframe);
  
  // 4. 设置返回值并跳转
  continuation->throw_value = val;
  SCM_I_LONGJMP (continuation->jmpbuf, 1);
}
```

**关键差异：**
- 检查 critical section，防止在不安全的情况下调用 continuation
- 使用 `scm_i_dowinds` 在 wind 过程中复制栈，确保 wind handlers 在正确的栈上执行
- 恢复调试框架信息
- 使用 `SCM_FLUSH_REGISTER_WINDOWS` 刷新寄存器窗口（某些架构需要）

#### 3. Continuation Barrier

Guile 1.8 实现了 `with-continuation-barrier`，通过 `continuation_root` 来限制 continuation 的作用域：

```c
SCM scm_i_with_continuation_barrier (...)
{
  // 建立新的 continuation root
  old_controot = thread->continuation_root;
  thread->continuation_root = scm_cons (thread->handle, old_controot);
  
  // 在 catch 中执行函数
  result = scm_c_catch (...);
  
  // 恢复旧的 continuation root
  thread->continuation_root = old_controot;
  return result;
}

static SCM continuation_apply (SCM cont, SCM args)
{
  // 检查 continuation root 是否匹配
  if (continuation->root != thread->continuation_root) {
    SCM_MISC_ERROR ("invoking continuation would cross continuation barrier: ~A",
                    scm_list_1 (cont));
  }
  // ...
}
```

### Guile 1.8 的优势

1. **完善的错误检查**：检查 critical section、continuation barrier
2. **调试支持**：保存和恢复调试框架，支持栈回溯
3. **指针调整**：通过 `offset` 处理栈内指针，确保指针有效性
4. **架构特定优化**：针对 ia64 等架构的特殊处理
5. **更精确的栈管理**：使用 `SCM_STACKITEM` 而不是 `long`
6. **GC 集成**：continuation 对象由 GC 管理，自动处理内存
7. **更完善的 wind 处理**：在 wind 过程中复制栈，确保正确性

## 对比总结

| 特性 | pscm_cc | Guile 1.8 |
|------|---------|-----------|
| 栈数据类型 | `long` | `SCM_STACKITEM` |
| 指针调整 | ❌ | ✅ (offset) |
| 调试框架 | ❌ | ✅ |
| Continuation Barrier | ❌ | ✅ |
| Critical Section 检查 | ❌ | ✅ |
| 架构特定优化 | ❌ | ✅ (ia64) |
| 内存管理 | 手动 malloc | GC 管理 |
| Wind 处理 | 简单实现 | 完善的实现 |
| 代码复杂度 | 简单 | 复杂但完善 |

## pscm_cc 的改进方向

### 1. 短期改进（高优先级）

#### 1.1 使用更精确的栈数据类型
- **问题**：当前使用 `long` 存储栈数据，在某些架构上可能不够精确
- **改进**：定义 `SCM_STACKITEM` 类型，根据架构选择合适的类型（通常是 `SCM` 或 `void*`）

#### 1.2 添加指针调整机制
- **问题**：栈中可能包含指向栈本身的指针，直接复制会导致指针失效
- **改进**：保存 `offset`，在恢复栈时调整栈内指针

```c
struct SCM_Continuation {
  // ...
  ptrdiff_t offset;  // 栈偏移量
  // ...
};

// 在复制栈后，需要遍历栈数据并调整指针
void adjust_stack_pointers(SCM_Continuation *cont, void *dst) {
  // 遍历栈数据，调整指向栈内的指针
  // 这是一个复杂的过程，需要知道哪些是指针
}
```

#### 1.3 添加 Critical Section 检查
- **问题**：在 critical section 中调用 continuation 可能导致不一致状态
- **改进**：添加检查机制

```c
extern int g_critical_section_level;

void scm_dynthrow(SCM *cont, SCM *args) {
  if (g_critical_section_level > 0) {
    fprintf(stderr, "continuation invoked from within critical section.\n");
    abort();
  }
  // ...
}
```

#### 1.4 改进栈增长策略
- **问题**：固定增长 100 个 `long` 可能不够灵活
- **改进**：根据需要的空间动态增长，或使用更智能的策略

```c
void grow_stack(SCM *cont, SCM *args) {
  // 计算需要增长的空间
  size_t needed = continuation->stack_len - current_stack_space();
  size_t growth_size = (needed > 100) ? needed + 100 : 100;
  long growth[growth_size];  // C99 VLA 或动态分配
  scm_dynthrow(cont, args);
}
```

### 2. 中期改进（中优先级）

#### 2.1 添加调试框架支持
- **问题**：无法进行栈回溯和调试
- **改进**：保存和恢复调试框架信息

```c
struct SCM_DebugFrame {
  // 调试信息
  const char *function_name;
  const char *file_name;
  int line_number;
  SCM_DebugFrame *prev;
};

struct SCM_Continuation {
  // ...
  SCM_DebugFrame *dframe;  // 保存的调试框架
  // ...
};
```

#### 2.2 实现 Continuation Barrier
- **问题**：无法限制 continuation 的作用域
- **改进**：实现 `with-continuation-barrier`

```c
SCM *scm_with_continuation_barrier(SCM *proc) {
  SCM *old_root = g_continuation_root;
  g_continuation_root = make_continuation_root();
  
  // 在 catch 中执行
  SCM *result = scm_catch_all(proc);
  
  g_continuation_root = old_root;
  return result;
}

void scm_dynthrow(SCM *cont, SCM *args) {
  // 检查 continuation root
  if (continuation->root != g_continuation_root) {
    error("continuation barrier violation");
  }
  // ...
}
```

#### 2.3 改进 Wind Chain 处理
- **问题**：当前的 wind chain 处理可能不够完善
- **改进**：参考 Guile 的实现，在 wind 过程中复制栈

### 3. 长期改进（低优先级）

#### 3.1 架构特定优化
- **问题**：某些架构（如 ia64）需要特殊处理
- **改进**：针对特定架构添加优化代码

```c
#ifdef __ia64__
  // 保存和恢复寄存器备份存储
  continuation->backing_store = save_register_backing_store();
#endif
```

#### 3.2 GC 集成
- **问题**：手动管理 continuation 内存
- **改进**：集成 GC，自动管理 continuation 对象

#### 3.3 性能优化
- **问题**：栈复制可能成为性能瓶颈
- **改进**：
  - 使用更高效的复制方法
  - 延迟复制（copy-on-write）
  - 栈压缩技术

## 结论

pscm_cc 的 continuation 实现是一个**精简但功能完整**的实现。它成功地实现了基本的 continuation 功能，包括 `call/cc` 和 `dynamic-wind` 支持。与 Guile 1.8 相比，它牺牲了一些高级特性（如调试支持、continuation barrier）来换取实现的简单性。

对于 pscm_cc 的目标（驱动 TeXmacs），当前的实现应该是足够的。但如果要进一步提升，建议优先实现：
1. 指针调整机制（避免潜在的 bug）
2. Critical section 检查（提高安全性）
3. 更精确的栈数据类型（提高可移植性）

这些改进将显著提高 continuation 实现的健壮性和可维护性。
