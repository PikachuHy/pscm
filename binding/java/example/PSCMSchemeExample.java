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
