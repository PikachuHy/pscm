# Continuation

这次基于 Register Machine 做了Continuation的简单支持。

具体做法把 Register Machine 中的栈和寄存器都拷贝一份，
等到需要 `apply continuation` 时，再把这些栈和寄存器放回到 Register Machine 中。
