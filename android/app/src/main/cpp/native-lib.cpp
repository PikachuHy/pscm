#include <jni.h>
#include <pscm/Scheme.h>
// #include <spdlog/spdlog.h>
#include <string>

extern "C" JNIEXPORT jstring JNICALL Java_dev_pscm_android_MainActivity_stringFromJNI(JNIEnv *env, jobject /* this
*/) {
  std::string hello = "Hello from C++";
  return env->NewStringUTF(hello.c_str());
}

extern "C" JNIEXPORT jlong JNICALL Java_dev_pscm_android_MainActivity_createScheme(JNIEnv *env, jobject /* this */) {
  auto scm = new pscm::Scheme();
  return (jlong)scm;
}

extern "C" JNIEXPORT jstring JNICALL Java_dev_pscm_android_MainActivity_evalSchemeCode(JNIEnv *env, jobject /* this
                                                                                                             */
                                                                                       ,
                                                                                       jlong scm, jstring code) {
  auto _scm = (pscm::Scheme *)scm;
  auto c_str = env->GetStringUTFChars(code, nullptr);
  auto len = env->GetStringUTFLength(code);
  std::string s = std::string((char *)c_str, len);
  auto ret = _scm->eval(s.c_str());
  return env->NewStringUTF(ret.to_string().c_str());
}