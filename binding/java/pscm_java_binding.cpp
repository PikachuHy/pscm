#include "pscm_c_api.h"
#include <jni.h>
#include <string>

extern "C" JNIEXPORT jlong JNICALL Java_dev_pscm_PSCMScheme_createScheme(JNIEnv *env, jobject /* this */) {
  auto scm = pscm_create_scheme();
  return (jlong)scm;
}

extern "C" JNIEXPORT jstring JNICALL Java_dev_pscm_PSCMScheme_evalSchemeCode(JNIEnv *env, jobject /* this
                                                                                                   */
                                                                             ,
                                                                             jlong scm, jstring code) {
  auto p = (void *)scm;
  auto c_str = env->GetStringUTFChars(code, nullptr);
  auto ret = pscm_eval(p, c_str);
  auto s = pscm_to_string(ret);
  return env->NewStringUTF(s);
}