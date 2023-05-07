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
