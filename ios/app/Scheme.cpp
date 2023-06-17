//
//  Scheme.cpp
//  demo
//
//  Created by PikachuHy on 2023/6/17.
//

#include "Scheme.hpp"
#include <iostream>
#include <pscm/Scheme.h>

class SchemeImpl {
public:
  pscm::Scheme scm_;
};

Scheme::Scheme() {
  impl_ = new SchemeImpl();
}

Scheme::~Scheme() {
  delete impl_;
}

std::string Scheme::eval(const char *code) {
  std::cout << "eval: " << code << std::endl;
  auto ret = impl_->scm_.eval(code);
  return ret.to_string();
}
