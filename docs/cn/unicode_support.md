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
