# 模块系统实现方案：参考 Guile 1.8

本文档描述在 `pscm_cc` 中实现模块系统的方案，参考 Guile 1.8 的设计。目标是：

- 提供与 Guile 1.8 兼容的模块 API；
- 实现模块的命名空间隔离和导入导出机制；
- 支持模块的延迟加载和解析；
- 兼容当前的类型系统和求值器结构。

## 1. 需求与语义

### 1.1 核心 API

参考 Guile 1.8 的模块系统，需要实现以下核心功能：

#### 模块定义
- `(define-module name [options ...])`：定义一个新模块
- `(current-module)`：获取当前模块
- `(set-current-module module)`：设置当前模块

#### 模块使用
- `(use-modules spec ...)`：使用其他模块的公共接口
- `(module-use! module spec)`：将模块添加到使用列表

#### 模块查询
- `(resolve-module name)`：解析模块名称，返回模块对象
- `(module-ref module symbol [default])`：从模块中获取变量
- `(module-bound? module symbol)`：检查模块中是否有绑定
- `(module-locally-bound? module symbol)`：检查模块本地是否有绑定

#### 模块导出
- `(define-public name value)`：定义并导出变量
- `(export symbol ...)`：导出符号列表
- `(re-export symbol ...)`：重新导出符号

#### 模块接口
- `%module-public-interface`：模块的公共接口（另一个模块对象）

### 1.2 Guile 1.8 模块系统设计要点

参考 `guile/libguile/modules.c` 和 `guile/ice-9/boot-9.scm`：

1. **模块数据结构**（参考 `boot-9.scm` 中的 `module-type`）：
   - `obarray`：哈希表，存储模块内的本地绑定（symbol -> variable）
   - `uses`：模块列表，从中继承非本地绑定
   - `binder`：可选的绑定过程，用于延迟绑定
   - `eval-closure`：查找策略函数，定义符号查找规则
   - `transformer`：语法转换器，用于宏展开
   - `name`：模块名称（列表形式，如 `(guile-user)`）
   - `kind`：模块类型（'module, 'interface, 'directory 等）

2. **符号查找策略**（参考 `modules.c` 中的 `module_variable`）：
   - 首先在模块的 `obarray` 中查找
   - 如果没找到，调用 `binder` 过程（如果存在）
   - 如果还没找到，在 `uses` 列表中的模块递归查找
   - 如果都没找到，返回 `#f`

3. **模块接口**：
   - 每个模块有一个公共接口（`%module-public-interface`）
   - 公共接口也是一个模块对象，只包含导出的绑定
   - `use-modules` 实际使用的是模块的公共接口

4. **当前模块管理**：
   - 使用 fluid（动态绑定）管理当前模块
   - `current-module` 返回当前 fluid 中的模块，如果没有则返回根模块

## 2. 设计方案（pscm_cc）

### 2.1 Module 类型表示

#### 2.1.1 类型定义

在 `pscm.h` 中新增：

```cpp
// 在 SCM::Type 枚举中新增
enum Type { 
  // ... 现有类型
  MODULE  // 模块类型
};

// 模块结构体
struct SCM_Module {
  SCM_HashTable *obarray;        // 本地绑定哈希表 (symbol -> variable)
  SCM_List *uses;                // 使用的模块列表
  SCM_Procedure *binder;         // 可选的绑定过程 (module symbol definep) -> variable | #f
  SCM_Procedure *eval_closure;   // 查找策略函数 (symbol definep) -> variable | #f
  SCM_Procedure *transformer;   // 语法转换器 (expr) -> expr
  SCM_List *name;                // 模块名称列表，如 (guile-user)
  SCM_Symbol *kind;              // 模块类型：'module, 'interface, 'directory
  SCM_Module *public_interface;  // 公共接口模块（指向另一个模块对象）
  SCM_List *exports;             // 导出的符号列表（用于公共接口）
};
```

#### 2.1.2 辅助函数

```cpp
// 类型检查
inline bool is_module(SCM *scm) {
  return scm->type == SCM::MODULE;
}

// 类型转换
inline SCM_Module *cast_module(SCM *scm) {
  if (!is_module(scm)) {
    type_error(scm, "module");
  }
  return static_cast<SCM_Module*>(scm->value);
}

// 创建模块
SCM *scm_make_module(SCM_List *name, int obarray_size = 31);
```

### 2.2 当前模块管理

#### 2.2.1 全局变量

在 `eval.cc` 或新建 `module.cc` 中：

```cpp
// 当前模块（使用简单的全局变量，未来可改为 fluid）
static SCM *g_current_module = nullptr;
static SCM *g_root_module = nullptr;  // 根模块（guile-user）

// 模块注册表：模块名称 -> 模块对象
static SCM_HashTable *g_module_registry = nullptr;
```

#### 2.2.2 当前模块操作

```cpp
// 获取当前模块
SCM *scm_current_module() {
  if (g_current_module) {
    return g_current_module;
  }
  // 如果没有当前模块，返回根模块
  if (!g_root_module) {
    // 初始化根模块
    g_root_module = scm_make_module(scm_list_from_strings({"guile-user"}), 31);
    g_current_module = g_root_module;
  }
  return g_root_module;
}

// 设置当前模块
SCM *scm_set_current_module(SCM *module) {
  if (!is_module(module)) {
    eval_error("set-current-module: expected module");
  }
  SCM *old = scm_current_module();
  g_current_module = module;
  return old;
}
```

### 2.3 模块查找策略

#### 2.3.1 变量查找

参考 `modules.c` 中的 `module_variable` 实现：

```cpp
// 在模块中查找变量
SCM *scm_module_variable(SCM_Module *module, SCM_Symbol *sym, bool definep) {
  // 1. 检查模块 obarray
  SCM *var = scm_hash_table_ref(module->obarray, wrap_sym(sym));
  if (var && !is_falsy(var)) {
    return var;
  }
  
  // 2. 调用 binder（如果存在且不是定义操作）
  if (!definep && module->binder) {
    SCM_List *args = scm_list_from_values({
      wrap_module(module),
      wrap_sym(sym),
      wrap_bool(false)
    });
    var = apply_procedure(/* env */, module->binder, args);
    if (var && !is_falsy(var)) {
      return var;
    }
  }
  
  // 3. 搜索 uses 列表
  SCM_List *uses = module->uses;
  while (uses) {
    SCM *use_module_scm = uses->data;
    if (is_module(use_module_scm)) {
      SCM_Module *use_module = cast_module(use_module_scm);
      var = scm_module_variable(use_module, sym, false);
      if (var && !is_falsy(var)) {
        return var;
      }
    }
    uses = uses->next;
  }
  
  return wrap_bool(false);  // #f
}
```

#### 2.3.2 模块解析

```cpp
// 解析模块名称，返回模块对象
SCM *scm_resolve_module(SCM_List *name) {
  // 1. 检查注册表
  if (g_module_registry) {
    SCM *module = scm_hash_table_ref(g_module_registry, wrap_list(name));
    if (module && is_module(module)) {
      return module;
    }
  }
  
  // 2. 尝试从文件系统加载（未来实现）
  // 这里先返回错误或创建新模块
  
  eval_error("resolve-module: module not found: %s", /* name string */);
}
```

### 2.4 define-module 实现

#### 2.4.1 特殊形式处理

在 `eval.cc` 中添加：

```cpp
// define-module 特殊形式
SCM *eval_define_module(SCM_Environment *env, SCM_List *l) {
  // 语法: (define-module name [options ...])
  if (!l || !l->data) {
    eval_error("define-module: missing module name");
  }
  
  SCM *name_scm = l->data;
  if (!is_list(name_scm)) {
    eval_error("define-module: module name must be a list");
  }
  
  SCM_List *name = cast_list(name_scm);
  
  // 解析选项（简化版，只处理基本选项）
  SCM_List *options = l->next;
  SCM_List *use_modules = nullptr;
  SCM_List *exports = nullptr;
  
  // 创建模块
  SCM *module_scm = scm_make_module(name, 31);
  SCM_Module *module = cast_module(module_scm);
  
  // 处理选项
  while (options) {
    // 简化处理，实际需要解析 #:use-module, #:export 等
    options = options->next;
  }
  
  // 注册模块
  if (!g_module_registry) {
    g_module_registry = scm_make_hash_table(61, SCM_HashTable::EQ);
  }
  scm_hash_table_set(g_module_registry, wrap_list(name), module_scm);
  
  // 设置当前模块
  scm_set_current_module(module_scm);
  
  return module_scm;
}
```

#### 2.4.2 注册特殊形式

在 `eval.cc` 的 `eval` 函数中添加：

```cpp
if (is_sym(car) && strcmp(sym->data, "define-module") == 0) {
  return eval_define_module(env, l);
}
```

### 2.5 use-modules 实现

#### 2.5.1 特殊形式处理

```cpp
// use-modules 特殊形式
SCM *eval_use_modules(SCM_Environment *env, SCM_List *l) {
  // 语法: (use-modules spec ...)
  SCM_Module *current = cast_module(scm_current_module());
  
  while (l) {
    SCM *spec_scm = l->data;
    // spec 可以是模块名称列表，如 (ice-9 common-list)
    if (is_list(spec_scm)) {
      SCM_List *name = cast_list(spec_scm);
      SCM *module_scm = scm_resolve_module(name);
      SCM_Module *module = cast_module(module_scm);
      
      // 获取模块的公共接口
      SCM_Module *interface = module->public_interface;
      if (!interface) {
        // 如果没有公共接口，使用模块本身
        interface = module;
      }
      
      // 添加到 uses 列表
      SCM_List *new_use = new SCM_List();
      new_use->data = wrap_module(interface);
      new_use->next = current->uses;
      current->uses = new_use;
    }
    l = l->next;
  }
  
  return wrap_nil();
}
```

### 2.6 模块导出

#### 2.6.1 define-public

```cpp
// define-public 特殊形式
SCM *eval_define_public(SCM_Environment *env, SCM_List *l) {
  // 语法: (define-public name value) 或 (define-public (name args ...) body ...)
  // 先调用普通的 define
  SCM *result = eval_define(env, l);
  
  // 然后导出符号
  SCM *name_scm = l->data;
  if (is_list(name_scm)) {
    // (name args ...) 形式
    name_scm = cast_list(name_scm)->data;
  }
  
  if (is_sym(name_scm)) {
    SCM_Symbol *sym = cast_sym(name_scm);
    scm_module_export(scm_current_module(), sym);
  }
  
  return result;
}
```

#### 2.6.2 export

```cpp
// export 特殊形式
SCM *eval_export(SCM_Environment *env, SCM_List *l) {
  SCM_Module *module = cast_module(scm_current_module());
  
  while (l) {
    SCM *sym_scm = l->data;
    if (is_sym(sym_scm)) {
      scm_module_export(module, cast_sym(sym_scm));
    }
    l = l->next;
  }
  
  return wrap_nil();
}

// 导出符号的辅助函数
void scm_module_export(SCM_Module *module, SCM_Symbol *sym) {
  // 添加到导出列表
  SCM_List *new_export = new SCM_List();
  new_export->data = wrap_sym(sym);
  new_export->next = module->exports;
  module->exports = new_export;
  
  // 更新公共接口
  scm_update_module_public_interface(module);
}
```

### 2.7 环境与模块的集成

#### 2.7.1 修改环境查找

在 `environment.cc` 中修改 `scm_env_search`：

```cpp
SCM *scm_env_search(SCM_Environment *env, SCM_Symbol *sym) {
  // 1. 先在词法环境中查找
  auto entry = scm_env_search_entry(env, sym);
  if (entry) {
    return entry->value;
  }
  
  // 2. 如果当前模块存在，在模块中查找
  SCM *current_mod = scm_current_module();
  if (current_mod && is_module(current_mod)) {
    SCM_Module *module = cast_module(current_mod);
    SCM *var = scm_module_variable(module, sym, false);
    if (var && !is_falsy(var)) {
      // 返回变量的值
      return scm_variable_ref(var);
    }
  }
  
  SCM_ERROR_SYMTBL("find %s, not found\n", sym->data);
  return nullptr;
}
```

#### 2.7.2 修改 define

在 `eval.cc` 中修改 `eval_define`：

```cpp
SCM *eval_define(SCM_Environment *env, SCM_List *l) {
  // ... 现有代码 ...
  
  // 如果当前模块存在，定义到模块中
  SCM *current_mod = scm_current_module();
  if (current_mod && is_module(current_mod)) {
    SCM_Module *module = cast_module(current_mod);
    // 在模块的 obarray 中创建绑定
    scm_hash_table_set(module->obarray, wrap_sym(sym), wrap_variable(value));
  } else {
    // 否则定义到环境中
    scm_env_insert(env, sym, value);
  }
  
  return wrap_nil();
}
```

### 2.8 初始化与接口暴露

#### 2.8.1 初始化函数

在 `init.cc` 或新建 `module.cc` 中：

```cpp
void init_modules() {
  // 创建根模块
  g_root_module = scm_make_module(
    scm_list_from_strings({"guile-user"}), 
    31
  );
  g_current_module = g_root_module;
  
  // 初始化模块注册表
  g_module_registry = scm_make_hash_table(61, SCM_HashTable::EQ);
  
  // 注册内置函数
  scm_define_function("current-module", 0, 0, 0, scm_c_current_module);
  scm_define_function("set-current-module", 1, 0, 0, scm_c_set_current_module);
  scm_define_function("resolve-module", 1, 0, 0, scm_c_resolve_module);
  scm_define_function("module-ref", 2, 1, 0, scm_c_module_ref);
  scm_define_function("module-bound?", 2, 0, 0, scm_c_module_bound_p);
  
  // 注册特殊形式（在 eval.cc 中处理）
  // define-module, use-modules, export, define-public 等
}
```

## 3. 实现步骤

### 阶段 1：基础数据结构
1. ✅ 在 `pscm.h` 中添加 `MODULE` 类型和 `SCM_Module` 结构体
2. ✅ 实现 `scm_make_module`、`is_module`、`cast_module` 等基础函数
3. ✅ 实现当前模块管理（全局变量 + 简单函数）

### 阶段 2：模块查找
1. ✅ 实现 `scm_module_variable`（查找策略）
2. ✅ 实现 `scm_resolve_module`（模块解析，简化版）
3. ✅ 实现模块注册表

### 阶段 3：模块定义和使用
1. ✅ 实现 `eval_define_module` 特殊形式
2. ✅ 实现 `eval_use_modules` 特殊形式
3. ✅ 测试基本的模块定义和使用

### 阶段 4：模块导出
1. ✅ 实现 `eval_export` 和 `eval_define_public`
2. ✅ 实现公共接口机制
3. ✅ 实现 `scm_module_export` 辅助函数

### 阶段 5：环境集成
1. ✅ 修改 `scm_env_search` 集成模块查找
2. ✅ 修改 `eval_define` 支持模块定义
3. ✅ 测试模块与环境的交互

### 阶段 6：完善功能
1. ⏳ 实现 `module-ref`、`module-bound?` 等查询函数
2. ⏳ 实现模块文件加载（与 `load` 集成）
3. ⏳ 实现模块路径搜索（`%load-path`）
4. ⏳ 错误处理和边界情况

## 4. 测试与兼容性

### 4.1 基本测试

```scheme
;; 测试模块定义
(define-module (test))
(display (current-module)) (newline)

;; 测试模块使用
(define-module (m1))
(define-public (hello) "hello")
(define-module (m2))
(use-modules (m1))
(display (hello)) (newline)
```

### 4.2 Guile 兼容性测试

参考 `test/module/` 目录下的测试文件，确保：
- `define-module` 语法兼容
- `use-modules` 行为一致
- `export` 和 `define-public` 正常工作
- 模块查找策略正确

## 5. 与后续扩展的关系

### 5.1 文件加载集成

模块系统需要与 `load` 函数集成：
- `load` 文件时，文件中的 `define-module` 应该创建模块
- 模块文件路径搜索（如 `(ice-9 common-list)` -> `ice-9/common-list.scm`）

### 5.2 宏系统集成

模块的 `transformer` 字段用于语法转换，未来可以：
- 支持模块级别的宏定义
- 支持 `use-syntax` 语法

### 5.3 性能优化

- 模块查找缓存
- 公共接口缓存
- 延迟绑定优化

## 6. 参考实现

- Guile 1.8 源码：
  - `guile/libguile/modules.c`：C 层实现
  - `guile/libguile/modules.h`：C 层接口
  - `guile/ice-9/boot-9.scm`：Scheme 层实现（模块类型定义、`define-module`、`use-modules` 等）

- 关键函数：
  - `scm_current_module`：获取当前模块
  - `scm_resolve_module`：解析模块
  - `module_variable`：模块变量查找
  - `process-define-module`：处理 `define-module` 表单
  - `process-use-modules`：处理 `use-modules` 表单

