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
  auto c_str = env->GetStringChars(code, nullptr);
  auto len = env->GetStringLength(code);
  pscm::UString s(c_str, len);
  auto ret = _scm->eval(s);
  auto s3 = ret.to_string();
  return env->NewString(static_cast<const jchar *>(s3.getBuffer()), s3.length());
}