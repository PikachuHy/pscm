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
- `STRING`: 字符串类型，目前只是在 `std::string` 上糊了一下
- `CHAR`: 字符类型，底下是一个 `std::string` ，因为一个 `char` 放不了中文字符，暂时也不想用 `wchar` 之类的
- `NUMBER`: 数类型，目前对 `int` 和 `float` 做了简单支持
- `SYMBOL`: 符号类型，表示 pscm 代码中的一个符号
- `PAIR`: pscm中的list是由一个又一个 `PAIR` 嵌套起来表达的
- `FUNCTION`: `C++` 函数类型，执行的是 `C++` 代码，暂不支持在这个函数内调用 pscm 的代码
- `PROCEDURE`: pscm 函数类型，该函数由 pscm 代码定义
- `MACRO`: 宏类型，目前还未支持了自定义的 lisp-style 宏
- `CONTINUATION`: 用于表达 `call/cc` 中的 continuation 概念，目前通过拷贝栈的方式实现了 continuation

