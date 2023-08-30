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
