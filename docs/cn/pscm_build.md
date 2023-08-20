# pscm-build

[pscm-build](https://github.com/PikachuHy/pscm/tree/master/tool/pscm-build) 是在 pscm 的基础上开发的一个构建系统，相关的概念借鉴 [bazel](https://bazel.build) 。

::: warning
pscm-build 依然处于非常简陋的状态
:::

pscm-build 是基于 pscm 开发的第一个工具，目前能够

- 实现构建 pscm
- 支持 C++20 Modules

## 设计目标

pscm-build 的目标是在 C++20 Modules 的背景下，探索 C++ 的构建系统和包管理器。

- 利用 C++20 Modules 加速 C++ 代码的构建
- 实现基于 C++20 Modules 包管理

## 当前的设计

- 完全基于 C++20 Modules 的代码风格编写
- 通过 `repo.pscm` 文件定位仓库根目录，pscm-build 会从调用命令的目录往上找，找到为止（找不到直接报错）
- 通过 `build.pscm` 划分包，一个仓库可以有多个包
- 通过 Label `@repo_name//package_name:target_name` 定位具体的某个 target
- 通过 Action 运行编译命令，子进程通过库 [subprocess](https://github.com/benman64/subprocess.git) 调用

## 运行要求

- clang 16及以上：默认采用构建 C++20 Modules 模式，需要对 C++20 Modules 支持比较完整的 LLVM 版本
- ccache: 由于目前没有实现 cache，默认使用 ccache 驱动 clang
- 设定环境变量 CC: 通过 CC 获取编译器路径。
  由于 AppleClang 目前还不支持 C++20 Modules,
  在 MacOS 上可以通过 `brew install llvm` 安装最新的 LLVM，
  然后通过 `export CC=/usr/local/opt/llvm/bin/clang` 使用。

## 规则简介

目前支持 `cpp_library`, `cpp_binary`, `cpp_test` 3个规则。
每个规则默认有一个 `name` 属性，用于标识当前 target。
此外，还支持

- `srcs`: 源码文件，支持使用 `glob`
- `hdrs`: 头文件，支持使用 `glob`
- `includes`: 库头文件的路径，会传递给依赖它的库/二进制程序。仅 `cpp_library` 支持
- `defines`: 库宏定义，会传递给依赖它的库/二进制程序。仅 `cpp_library` 支持
- `copts`: 编译时的flags
- `deps`: 依赖的库

## 样例代码
- 创建一个库
```scheme
(cpp_library
 (name "pscm")
 (srcs
  (glob "src/*.cpp"))
 (hdrs
  (glob "include/**/*.h"))
 (includes "include")
 (copts "-std=c++20" "-I" "build/generate")
 (deps ":spdlog"))
```
- 创建一个二进制程序
```scheme
(cpp_binary
  (name "pscm-main")
  (srcs "main.cpp")
  (deps ":pscm"))
```
- 创建一个测试
```scheme
(cpp_test
  (name "r4rs_test")
  (srcs (glob "test/r4rs/*.cpp"))
  (deps ":pscm" ":doctest")
  (copts "-std=c++20"))
```

- 支持 C++20 Modules

pscm-build 会自动辨别 Module Interface (通过 clang-scan-deps 的扫描结果)，
无需手动指定

```scheme
(cpp_binary
  (name "main")
  (srcs (glob "*.cc" "*.cppm" "*.cpp"))
  (copts "-std=c++20"))
```

- 构建所有的 target

```shell
export CC=/usr/local/opt/llvm/bin/clang
pscm-build build :all
```

默认构建产物在仓库根目录下的 `pscm-build-bin` 目录

- 删除构建目录

```shell
pscm-build clean
```

## 未来计划

- cache
- action graph
- sandbox
