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
