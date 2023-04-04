//
// Created by PikachuHy on 2023/4/5.
//
#include <iostream>
#include <pscm/Scheme.h>
#include <spdlog/spdlog.h>
using namespace pscm;

int main() {
  spdlog::set_level(spdlog::level::err);
  Scheme scm;
  auto version = scm.eval("(version)");
  std::cout << "Welcome to PikachuHy's Scheme" << std::endl;
  std::cout << "version: " << version << std::endl;
  return 0;
}