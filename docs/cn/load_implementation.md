## 文件加载函数实现（`load` / `primitive-load`）

✅ **已实现**：文件加载功能已完整实现，支持 `load` 和 `primitive-load`，并与模块系统集成。

本文档描述 `pscm cc` 中文件加载相关函数的实现，目标是：

- 提供与 R4RS 兼容的 `(load filename)` 行为，用于在当前环境中顺序加载并执行 Scheme 文件；
- 提供与 Guile 1.8 兼容的底层接口 `primitive-load`，支持模块系统和加载路径。

### 1. 需求与语义

- **基本接口**
  - `load filename`：其中 `filename` 为字符串，表示要加载的 Scheme 源文件路径。
  - `primitive-load filename`：底层加载函数，语义与 `load` 基本一致，但更偏向内部使用和与 Guile 行为对齐。
- **行为要求**
  - 使用已有的 `parse_file` 将文件解析为 AST 列表；
  - 在“当前顶层环境”中，按顺序对每个表达式调用 `eval`，返回最后一个表达式的值；
  - 当文件不存在或解析失败时，抛出错误（`eval_error`），并带有清晰的错误信息。

### 2. 与现有基础设施的关系

- **解析层**
  - 已存在 `SCM_List *parse_file(const char *filename);`（定义于 `parse.cc`，声明在 `pscm.h`），可以直接复用，将文件解析为表达式链表。
- **求值层**
  - 已存在 `SCM *eval_with_env(SCM_Environment *env, SCM *data);` 以及便捷函数 `eval_with_list`；
  - 顶层求值使用全局环境 `g_env`（`main.cc` 中 `do_eval` 已经展示了“parse + 逐表达式求值”的模式）。
- **端口与 I/O**
  - 当前 R4RS 测试中，`load-test-obj` 只是用来写入到临时文件并再读回（通过 `read` / `open-input-file` 等），因此 `load` 实现只需负责“文件 → AST → eval”，不需要依赖端口层 API。

### 3. 设计方案

#### 3.1 C++ 侧实现形式

- 在 C++ 侧新增一个实现函数：
  - `SCM *scm_c_primitive_load(SCM *filename);`
    - 参数：`filename` 必须是字符串，表示文件路径；
    - 行为：
      1. 检查类型，不是字符串则报错：`eval_error("primitive-load: expected string");`
      2. 将 `SCM_String` 中的 `data` 作为 C 字符串传入 `parse_file`；
      3. 若 `parse_file` 返回 `nullptr`，说明文件读取或解析失败，调用 `eval_error("primitive-load: failed to load file: %s", s->data);`
      4. 遍历解析得到的 `SCM_List`，在“顶层环境”中依次 `eval_with_env`，保存最后一个结果并返回。
    - 顶层环境选择：
      - 与 `port.cc` 中 `call-with-input-file` 的实现保持一致，使用 `g_env.parent ? g_env.parent : &g_env` 作为当前交互环境。

- 在初始化阶段注册 Scheme 函数：
  - `scm_define_function("primitive-load", 1, 0, 0, scm_c_primitive_load);`
  - `scm_define_function("load", 1, 0, 0, scm_c_primitive_load);`
  - ✅ **已实现**：`load` 和 `primitive-load` 已注册，并支持 `%load-path` 模块搜索路径

#### 3.2 错误处理与源位置

- `parse_file` 已负责为 AST 注入源位置信息（文件名、行列号）；
- `eval_with_env` 在执行过程中若遇到错误，会通过 `eval_error` 报告，并利用当前表达式的 `source_loc` 形成调用栈；
- `primitive-load` 中的显式错误（如文件不存在、解析失败）通过 `eval_error` 报告文件名，便于用户定位问题。

### 4. 与 CLI 行为的对齐

- 当前 `pscm_cc` 的 CLI 已支持：
  - `--test FILE` / `-s FILE`：在 C++ 侧调用 `do_eval`，使用 `parse_file + my_eval` 按顺序执行文件；
- 新增的 `primitive-load` / `load` 主要用于 Scheme 代码内部动态加载其他 Scheme 文件：
  - 行为模式与 `do_eval` 类似，但暴露为内置过程；
  - R4RS 测试中的 `(load "tmp1")` 等调用依赖该能力。

### 5. 兼容性与后续扩展

- **Guile 兼容性**
  - 保留与 Guile 类似的命名：`primitive-load` 作为基础实现，`load` 作为用户接口；
  - 后续可以在 Scheme 层实现：
    - `%load-path`：搜索目录列表；
    - 处理相对路径与模块路径。

- **模块系统集成** ✅ **已实现**
  - `primitive-load` 作为"从文件加载并 eval"的底层原语，已与模块系统集成；
  - 模块加载（如 `use-modules`）通过 `scm_resolve_module` 自动从文件系统加载模块文件；
  - 支持 `%load-path` 变量配置模块搜索路径，在根模块（`pscm-user`）中定义

### 6. 实现状态

✅ **已完成**：
- `load` 和 `primitive-load` 函数已实现并注册
- 支持文件解析和顺序求值
- 错误处理包含清晰的错误信息
- 与模块系统集成，支持 `%load-path` 路径搜索
- 模块文件自动加载（通过 `scm_resolve_module`）

### 7. 测试策略

- 使用命令运行 R4RS 测试：
  - `./pscm_cc --test test/r4rs/r4rstest.scm`
- 重点验证：
  - ✅ `(load "tmp1")` 正常工作，不再出现"symbol 'load' not found"
  - ✅ 加载的文件中定义的变量能在后续表达式中被正常引用
  - ✅ 若传入不存在的文件名，会得到清晰的错误信息
  - ✅ 模块文件可以通过 `use-modules` 自动加载


