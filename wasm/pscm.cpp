//
// Created by PikachuHy on 2023/3/11.
//
#include <emscripten/emscripten.h>
#include <iostream>
#include <pscm/Scheme.h>
#include <spdlog/spdlog.h>
using namespace pscm;
#ifdef __cplusplus
extern "C" {
#endif
void EMSCRIPTEN_KEEPALIVE foo() {
  std::cout << "hello" << std::endl;
}

int EMSCRIPTEN_KEEPALIVE bar(int val) {
  std::cout << "hello: " << val << std::endl;
  return val;
}

void *EMSCRIPTEN_KEEPALIVE create_scheme(bool use_register_machine) {
  if (use_register_machine) {
    std::cout << "create scheme with register machine" << std::endl;
  }
  else {
    std::cout << "create scheme" << std::endl;
  }
  return new Scheme(use_register_machine);
}

void EMSCRIPTEN_KEEPALIVE destroy_scheme(void *scm) {
  std::cout << "destroy scheme" << std::endl;
  auto p = (Scheme *)scm;
  delete p;
}

const char *EMSCRIPTEN_KEEPALIVE eval(void *scm, const char *code) {
  auto p = (Scheme *)scm;
  std::cout << "eval: " << code << std::endl;
  auto ret = p->eval(code);
  std::cout << "--> " << ret << std::endl;
  auto s = ret.to_string();
  char *str = new char[s.size() + 1];
  std::memcpy(str, s.data(), s.size());
  str[s.size()] = '\0';
  return str;
}
#ifdef __cplusplus
};
#endif
int EMSCRIPTEN_KEEPALIVE main() {
  spdlog::set_level(spdlog::level::err);
  std::cout << "Start pscm-v2" << std::endl;
  std::cout << "With PSCM" << std::endl;
  return 0;
}