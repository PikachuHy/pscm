package dev.pscm.android;

import android.os.Bundle;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
  private long scm;
  static {
    System.loadLibrary("app");
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    // Example of a call to a native method
    TextView tv = (TextView)findViewById(R.id.sample_text);
    tv.setText(stringFromJNI());
    scm = createScheme();
    String ret = evalSchemeCode(scm, "(version)");
    tv.setText(ret);
  }

  /**
   * A native method that is implemented by the 'native-lib' native library,
   * which is packaged with this application.
   */
  public native String stringFromJNI();
  public native long createScheme();
  public native String evalSchemeCode(long scm, String code);
}
