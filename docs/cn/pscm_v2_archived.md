# pscm v2 文档汇总(归档)

代码在 [v2-legacy](https://github.com/PikachuHy/pscm/tree/v2-legacy) 分支

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
- 基本通过 r4rstest.scm 测试
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

# Build PikachuHy's Scheme

## Preparation

on MacOS

```
brew install cmake
brew install ninja
brew install pkg-config
brew install icu4c
```

### Required toolchain

if you build other projects under pscm repo

```shell
sudo apt-get install -y build-essential
sudo apt-get install -y libsdl2-dev libsdl2-ttf-dev libsdl2-image-dev libsdl2-mixer-dev
```

### Required git submodule

```shell
git submodule update --init
```

### Required ninja (version >= 1.11.0)

if you build pscm with C++20 Modules

Ref: https://github.com/ninja-build/ninja/releases

## CMake (Recommanded)

```shell
mkdir build && cd build
cmake ..
make -j
```

## xmake

目前Windows下，推荐使用xmake进行构建。xmake可以管理icu等依赖，无需同步git子模块

```shell
xmake config --yes --mode=Release
xmake build --jobs=4
```

## Bazel

```shell
bazel build :all
```

- build pscm android app

set `ANDROID_HOME` and `ANDROID_NDK_HOME`

```
# for example
# export ANDROID_HOME=$HOME/Library/Android/sdk
# export ANDROID_NDK_HOME=$HOME/Library/Android/sdk/ndk/25.1.8937393
bazel build //android/app/src/main:app --android_crosstool_top=@androidndk//:toolchain --fat_apk_cpu=arm64-v8a
```

- install pscm android app

```
adb install bazel-bin/android/app/src/main/app.apk
```

Note: don't
use `bazel mobile-install //android/app/src/main:app --android_crosstool_top=@androidndk//:toolchain --fat_apk_cpu=arm64-v8a`,
which may cause app crash

- launch pscm android app

```
adb shell am start -n dev.pscm.android/dev.pscm.android.MainActivity
```

- build pscm ios app

```
bazel build //ios/app:ios-app
```

- run pscm ios app on simulator

```
bazel run //ios/app:ios-app
```
## pscm-build (WIP)

```shell
pscm-build
```


# Cell

`Cell` 在pscm中表达scheme object的概念。
所有在pscm代码中出现的事物都是用 `Cell` 来表示的。

```cpp
class Cell {
  Tag tag_ = Tag::NONE;
  void *data_ = nullptr;
};
```

`Cell` 由 `Tag` 和 `data` 两个部分组成。
`Tag` 表明具体的数据类型，`data` 指向这个具体数据的地址。

⚠️：关于 `Cell` 是用指针还是用值，这个是暂时没有想清楚的，所以暂时不管。

目前 `Cell` 中存储的数据类型有

- `NONE`: 默认值，对应scheme中的 unspecified
- `EXCEPTION`: 暂时用来存放C++代码中出现的异常
- `NIL`: 空列表
- `BOOL`: 布尔值, `true` or `false`
- `STRING`: 字符串类型，内部采用ICU4C框架的`icu::UnicodeString`类
- `CHAR`: 字符类型，底下是一个 `UChar32` 
- `NUMBER`: 数类型，目前对 `int` 和 `float` 做了简单支持
- `SYMBOL`: 符号类型，表示 pscm 代码中的一个符号
- `PAIR`: pscm中的list是由一个又一个 `PAIR` 嵌套起来表达的
- `FUNCTION`: `C++` 函数类型，执行的是 `C++` 代码，暂不支持在这个函数内调用 pscm 的代码
- `PROCEDURE`: pscm 函数类型，该函数由 pscm 代码定义
- `MACRO`: 宏类型，目前还未支持了自定义的 lisp-style 宏
- `CONTINUATION`: 用于表达 `call/cc` 中的 continuation 概念，目前通过拷贝栈的方式实现了 continuation

# Equivalence predicates

`eq?`, `eqv?`和`equal?`在不同的场合都有使用。
本文档总结在`pscm`中三者的实现方法。
在`pscm`中，通过`Cell`表达一个Scheme的类型。
`Cell`内部有一个`void*`指向具体类型地址。
```cpp
class {
    void* data_;
}
```
## eq?
根据R5RS文档的描述
> eq? is the finest or most discriminating

## eqv?
`eqv?`稍微松一点，对于`Symbol`, `Number`, `Char`和`String`类型，会比较具体的值。
其他类型同`eq?`。

## equal?
`equal?`是最松的判定，对`Pair`类型，也会判定`car`和`cdr`是否是`eqv?`。

举例说明
```scheme
(equal? '(a) '(a))
;;; ==> #t
(eq? '(a) '(a))
;;; ==> #f
```

# Register Machine

Register Machine 是参考 SICP 实现的，
具体在第五章和b站视频上都有讲。

- https://www.bilibili.com/video/BV1Xx41117tr?p=17

# Continuation

这次基于 Register Machine 做了Continuation的简单支持。

具体做法把 Register Machine 中的栈和寄存器都拷贝一份，
等到需要 `apply continuation` 时，再把这些栈和寄存器放回到 Register Machine 中。

# R4RS Support

pscm 目前基本通过了 [r4rstest.scm](https://github.com/PikachuHy/pscm/blob/master/test/r4rs/r4rstest.scm)，

## 为什么选R4RS测试

- R4RS 足够简单
- 之前在 PicoScheme 上折腾了很久的 R4RS 测试
- R4RS 没有规定宏（附录有），暂时不想引入宏

## 对标准不同的实现

- 符号是区分大小写的，R4RS不区分
- 有些地方标准规定的是 `unspecified` ，最后都按照 guile 1.8 的结果实现
## 已知问题

- `(test-delay)` 没有通过，暂时放弃
- `(test-sc4)` 没有通过，暂时放弃
- 原测试读文件的部分，有一个错误，最后改了 expected 的值。按照和 guile 1.8 一样的读取结果处理

# Unicode Support

pscm 现在通过 icu4c 支持了 Unicode。

## 为什么选 icu4c

see https://github.com/PikachuHy/pscm/issues/8

## 如何构建 icu4c

如果不是交叉编译的情况，参考官方文档 [Building ICU4C](https://unicode-org.github.io/icu/userguide/icu4c/build.html) 即可。

交叉编译（比如在MacOS上）时，Android 和 WASM 两个版本使用官方的构建方法，我都没有正确构建出来。

目前 pscm 基于 Bazel 构建时，采用源码构建的方式，已经适配了 MacOS, Linux, Android, ios, WASM 这几个版本，代码在 [icu.bazel](https://github.com/PikachuHy/icu.bazel)。
交叉编译最难的是构建 icudata，目前采用的方式是将数据文件转换成 C 语言源码，然后编译。


## 如何使用 icu4c

使用 Bazel 时，icu4c 会从源码编译。

使用 xmake 时，icu4c 由其包管理器提供。

使用 CMake 时，需要先通过系统包管理器安装 icu4c，然后通过 pkgconfig 使用。

注意：WASM 通过 `-sUSE_ICU=1` 使用的 icu4c 没有构建 icudata，会导致无法得到正确的结果。(可以编译通过，但是运行出错)


## 参考

- https://github.com/google/zetasql/blob/master/bazel/icu.BUILD
- https://github.com/dio/icuuc
- https://github.com/qzmfranklin/icu
- https://github.com/NanoMichael/cross_compile_icu4c_for_android
- https://github.com/patrickgold/icu4c-android
- https://github.com/couchbaselabs/icu4c-android
- https://github.com/mabels/wasm-icu

# Language Binding

## C API

添加下面的基本 API

- `void *pscm_create_scheme();`
  创建一个Scheme解释器

- `void pscm_destroy_scheme(void *scm);`
  销毁解释器 `scm`

- `void *pscm_eval(void *scm, const char *code);`
  使用解释器 `scm` 对代码 `code` 进行求值

- `const char *pscm_to_string(void *value);`
  将所求的结果转换为字符串。注意，这里会拷贝一份字符串。

## Java API

通过类 `dev.pscm.PSCMScheme` 包裹 C API

- 使用前需要加载动态库。Android环境下需要直接依赖`//binding/java:pscm_java_binding`，其他环境加载`pscm-jni`
- 目前提供 `String PSCMScheme.eval(String code)` 接口，完成求值

使用示例

```java
import dev.pscm.PSCMScheme;
public class PSCMSchemeExample {

  static {
    // load shared library
    System.loadLibrary("pscm-jni");
  }

  public static void main(String args[]) {
    System.out.println("Hello World");
    PSCMScheme scm = new PSCMScheme();
    // eval (version)
    String ret = scm.eval("(version)");
    System.out.println(ret);
  }

}
```

## Python API

- 使用 `pybind11` 提供 `pypscm.Scheme` 类

使用示例

```python
import pypscm

scm = pypscm.Scheme()
print(scm)
ret = scm.eval("(+ 2 6)")
print(ret)
```

# pscm core

pscm core 是重新写的一套PikachuHy's Scheme实现。
从 core 这个名字也可以看出，我希望它足够的小，但有足够的有用。

## Why

之前做的实现，在支持 Continuation 和 Unicode 后，变得比较难演进。
所以我想再写一个更小的实现，不再拘泥 R5RS 标准。
我对它的定位更多是一个前缀表达式的语言。

## 设计目标

- 学习 LLVM
- 将 Scheme 编译到 LLVM IR，然后用 LLVM JIT运行。(也可以考虑直接出一个可执行文件)
- 强类型

::: warning
pscm 依然处于非常简陋的状态
:::

当前的版本

以下特性已实现编译到 LLVM IR，并使用 LLVM JIT运行。

- 基于 LLVM 17 开发
- 支持整型 `integer`，和整型数组 `array<integer>`
- 支持的操作： `map`, `+`, `-`, `>`, `<`
- 支持简单的函数定义和调用。函数在调用时才做类型检查和代码生成。只定义，不调用不会有任何的效果。

## 整体设计

不再使用之前基于 `Cell` 的实现，现在使用大量的指针和类继承。
基本类型是 `Value`，然后会把各种 `Value` 编译到 `AST`。
这里相当于前端的语法是前缀表达式，后面的流程和传统的 `C/C++` 差不多。

Codegen 时，把各种 `AST` 生成 LLVM IR，接着就可以使用 LLVM JIT执行LLVM IR。

```
Cell -> Value -> AST -> LLVM IR
```

目前依旧复用了老的pscm的parser，暂时没有重新写一套，所以刚拿到的依旧是 `Cell`。

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

# Krabby

一个简单的打字软件.开发代号“大钳蟹”
![Krabby](http://s1.52poke.wiki/wiki/thumb/a/a7/098Krabby.png/300px-098Krabby.png)
## 大钳蟹简介
大钳蟹（日文︰クラブ，英文︰Krabby）是水属性宝可梦。

外貌
大钳蟹有点像招潮蟹，大钳蟹的体色由红色和白色组成，头部的上半部分和两只蟹爪为红色，其余部分为白色。头顶有两个突起，下颌部分有着状似牙齿的凸起。六条肢体（包含钳子）都分节，它似乎还可以吃。

性别差异
大钳蟹没有性别差异。

特殊能力
性情
落日时会聚集在一起吐泡沫。

栖息地
沙滩的洞中。

## v3

基于SDL2的wasm版本正在开发中

![Krabby base on WASM SDL2](http://cdn.pikachu.net.cn/project/Krabby/krabby_v3_sdl2_wasm_screenshot.png)

## v2

### v2.1.0 支持统计
![Krabby v2.1.0 MacOS](http://cdn.pikachu.net.cn/project/Krabby/krabby_v2.1.0_macos_screenshot.png)

- 调整部分图标
- 支持统计打字速度
- 支持随机键盘练习
- 支持隐藏/显示键盘

### v2.0.0 支持MacOS
![Krabby v2.0.0 MacOS](http://cdn.pikachu.net.cn/project/Krabby/krabby_v2.0.0_macos_screenshot.png)

![Krabby v2.0.0 WASM](http://cdn.pikachu.net.cn/project/Krabby/krabby_v2.0.0_wasm_screenshot.png)

- 支持MacOS，移除对DTK的依赖
- 支持键盘提示
- 调整部分图标
- 基于Qt6.5部分支持WASM

## 金山打字通deepin版

- 打字练习的UI界面保持和windows版本一致

![1554212082558](https://img-blog.csdnimg.cn/20190403215627887.png)

- 图标从简，只显示文字
  - 时间
  - 速度
  - 进度
  - 正确率
  - 重置
  - 暂停
- 课程选择，使用下拉列表框
- 去掉其他的部分
- 打字的部分实现
  - 仅支持英文
  - 需要打的字为黑色，打字正确变成灰色，错误变成红色，都需要变
  - 自动换行
  - 每篇文章分成多个页，不同页之前不能干扰
  - 每页有5行输入的文本
  
### 当前实现
![demo](http://cdn.pikachu.net.cn/project/Krabby/krabby_v1_demo.png)

![score](http://cdn.pikachu.net.cn/project/Krabby/krabby_v1_score.png)

![article](http://cdn.pikachu.net.cn/project/Krabby/krabby_v1_article.png)

![setting](http://cdn.pikachu.net.cn/project/Krabby/krabby_v1_settings.png)
