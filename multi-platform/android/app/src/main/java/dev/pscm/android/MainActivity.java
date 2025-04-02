package dev.pscm.android;

import android.os.Bundle;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import dev.pscm.PSCMScheme;

public class MainActivity extends AppCompatActivity {
  private PSCMScheme scm;

  static {
    System.loadLibrary("app");
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    scm = new PSCMScheme();
    TextView tv = (TextView)findViewById(R.id.sample_text);
    String ret = scm.eval("(version)");
    tv.setText(ret);
  }
}
