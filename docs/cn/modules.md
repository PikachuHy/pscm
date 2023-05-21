# Modules Support

为了最终能够驱动TeXmacs，pscm目前的Modules实现方案几乎完全按照guile api设计与实现。

具体参考 [Guile Modules](https://www.gnu.org/software/guile/manual/html_node/Modules.html)

## API

- `use-modules`
accepts one or more interface specifications and, upon evaluation, arranges for those interfaces to be available to the current module.


