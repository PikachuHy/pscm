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
