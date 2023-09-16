# pscm 简介

pscm 是 PikachuHy's Scheme 的缩写，
是名字的第一个字母和 scheme 源码文件的后缀结合后造的一个词。

在改名为 pscm 之前，曾在 [PicoScheme](https://github.com/arichel/PicoScheme) 上做了一些实验，
但是都失败了。如果把 原来的 PicoScheme 称为 `v0` 的话，目前的这个 pscm 可以称为 `v2`，因为 `v1` 已经失败了。

## 设计目标

- 学习 SICP
- 学习 LLVM
- 把MSVC版本的TeXmacs跑起来
- 在浏览器上（WebAssembly）运行

::: warning
pscm 依然处于非常简陋的状态
:::

当前的版本

- 通过 SICP 中的 Register Machine 结合拷贝栈的方式简单实现了 continuation 。
- 通过 EMSDK 封装 C 接口，可以在浏览器上运行
- 基本通过 r4rstest.scm 测试，详见 [R4RS Support](r4rs_support.md)
- `map` 和 `for-each` 参数最多 2 个，有需求再增加参数个数

接下来需要探索的事情是如何对接 LLVM 的能力，将 pscm 代码编译成二进制文件。

## 整体设计

目前 pscm 支持两种解释执行的方式，一种是遍历AST （后面就称为DIRECT mode吧），另外一种是通过 Register Machine 。

遍历AST的方法基本等同原来的 [PicoScheme](https://github.com/arichel/PicoScheme) ，
由于这种做法我已经折腾很久了，暂时不是很想动。

SICP 中的 Register Machine 这个方法，我也折腾了挺久的。
由于之前的版本没有很好的实现 continuation ，这次特地花了不少时间去弄这个特性。
目前这个方法类似于生成逻辑上的指令，然后在一个巨大的 `switch-case` 循环中把这些指令执行掉。
没有生成具体的指令是因为之前的版本，我生成指令后，调试比较麻烦。
当前 pscm 中有各种 C++ 宏，这些都是为了调试方便点。

|                  | Continuation | JIT | AOT |
|------------------|--------------|-----|-----|
| DIRECT           | ❌            | ❌   | ❌   |
| Register Machine | ✅            | ❌   | ❌   |

## 参与开发

我目前只做了 continuation 的简单实现，其他的因为我之前已经折腾过一遍了。
暂时不会去弄，如果你感兴趣，欢迎提PR。

pscm 不局限于 Scheme 语言/语法，它更多是我学习编译原理的一个工具。
如果你想实验其他的功能，欢迎提PR。

目前GitHub的环境我基本搭好了。
如果你喜欢某些功能，但是 pscm 又没有实现。
你可以提Issue，如果我觉得比较好弄的话，周末会抽时间做一个简单的实现。

## Q & A

- 内存问题

  暂时没管，有意向使用 [Boehm garbage collector](https://github.com/ivmai/bdwgc)

- 代码风格

  通过GitHub CI就可以了

- 什么时候开始弄MSVC版本的TeXmacs

  预计对接完LLVM后，会考虑这个事情。具体得看情况

- 为什么用C++编写pscm

  `v0`采用的是C++，而作者熟悉原来的代码，所以没有换语言。

- Exception occurred here: evaluator terminate due to reach max step

  我做了一个最大计算步的设计，主要是为了debug实现错误死循环的情况。
  当然，对于本身就不会结束的程序（比如阴阳谜题），跑到一定的step就会抛异常停止。