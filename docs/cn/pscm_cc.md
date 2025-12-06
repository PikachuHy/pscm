# pscm cc

pscm cc 是重新写的一套PikachuHy's Scheme的实现。
这个版本最大的特性是学习类似Guile 1.8的方式，基于`longjmp/setjmp`实现了continuation。
代码规模大概在2000行。

cc后缀表示使用C++编写。实际上，这版本实现是使用尽可能少的C++特性来实现，目前用到了模板，重载等。

另外一个特殊点是，开发过程中基于llvm lit工具封装了pscm-lit做测试，可以直接写scm文件，然后检查求值结果是不是正确，非常方便。

paser部分复用了老的paser，后面会重写。

::: warning
pscm 依然处于非常简陋的状态
:::

## 设计目标

- 利用有限的C++特性实现一个精简版本的Guile 1.8，只保留驱动TeXmacs时的必要的特性。

## 整体设计

- 所有的类型都是`struct SCM`(内部是`void*`)。通过`cast<Type>(scm)`可以将`struct SCM`类型转换为具体的`Type`类型；通过`wrap(type)`可以将具体的`Type`类型转换为`struct SCM`(数字类型暂不支持)。
- List直接用链表实现，不再是pair套pair，操作时更加简便。
- eval部分，用了goto，在部分情况下可以减小栈的深度，做到类似尾递归的效果。
- C/C++代码注册到Scheme中采用类似Guile 1.8的接口，方便后续在TeXmacs中使用。

## TODO
- 后续会使用AI来重写代码，目前实现的部分完全是手搓的。
- parser重写
