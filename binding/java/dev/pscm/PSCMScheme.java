package dev.pscm;

public class PSCMScheme {

  private long scm = 0;

  public PSCMScheme() {
    init();
  }

  public void init() {
    if (scm == 0) {
      scm = createScheme();
    }
  }

  public String eval(String code) {
    return evalSchemeCode(scm, code);
  }

  /**
   * A native method that is implemented by the 'native-lib' native library,
   * which is packaged with this application.
   */
  public native long createScheme();
  public native String evalSchemeCode(long scm, String code);
}