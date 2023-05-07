# pscm-build

[pscm-build](https://github.com/PikachuHy/pscm/tree/master/tool/pscm-build) 是在 pscm 的基础上开发的一个构建系统，相关的概念借鉴 [bazel](https://bazel.build) 。

::: warning
pscm-build 依然处于非常简陋的状态
:::

pscm-build 是基于 pscm 开发的第一个工具，目前能够实现构建 pscm 。

## 规则简介

目前支持 `cpp_library`, `cpp_binary`, `cpp_test` 3个规则。
每个规则默认有一个 `name` 属性，用于标识当前 target。
此外，还支持

- `srcs`: 源码文件，支持使用 `glob`
- `hdrs`: 头文件，支持使用 `glob`
- `includes`: 库头文件的路径，会传递给依赖它的库/二进制程序。仅 `cpp_library` 支持
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
- 构建所有的 target

```shell
pscm-build :all
```
## 未来计划

- C++20 Modules support
- cache
- action graph
- sandbox
