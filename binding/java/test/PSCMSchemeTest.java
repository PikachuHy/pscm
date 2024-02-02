package test;
import static org.junit.Assert.assertEquals;

import dev.pscm.PSCMScheme;
import org.junit.Test;

public class PSCMSchemeTest {

  static {
    System.loadLibrary("pscm-jni");
  }

  //   public static void main(String args[]) {
  //     System.out.println("Hello World");
  //     PSCMScheme scm = new PSCMScheme();
  //     String ret = scm.eval("(version)");
  //     System.out.println(ret);
  //   }

  @Test
  public void testAdd() {
    PSCMScheme scm = new PSCMScheme();
    String ret = scm.eval("(+ 2 3)");
    assertEquals("should return 5 when calculate (+ 2 3)", "5", ret);
  }
}