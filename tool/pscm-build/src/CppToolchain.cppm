//
// Created by PikachuHy on 2023/8/16.
//
module;
#include <pscm/Logger.h>
#include <pscm/common_def.h>
#include <stdlib.h>
export module pscm.build:CppToolchain;
import std;
import pscm.logger;
import fmt;
import pscm.compat;
using namespace std::string_literals;

export namespace pscm::build {

class CppToolchain {
public:
  CppToolchain(std::string cc, std::string ar, std::string deps_scanner)
      : cc_(std::move(cc))
      , ar_(std::move(ar))
      , deps_scanner_(std::move(deps_scanner)) {
  }

  const std::string& cc() const {
    return cc_;
  }

  const std::string& ar() const {
    return ar_;
  }

  const std::string& deps_scanner() const {
    return deps_scanner_;
  }

private:
  std::string cc_;
  std::string ar_;
  std::string deps_scanner_;
};

CppToolchain init_cpp_toolchain() {
  PSCM_INLINE_LOG_DECLARE("pscm.build.toolchain");
  auto cc = getenv("CC");
  if (!cc) {
    PSCM_ERROR("CC not found");
    std::exit(1);
  }
  auto toolchain_path = fs::path(cc).parent_path();
  auto ar = toolchain_path.c_str() + "/llvm-ar"s;
  auto deps_scanner = toolchain_path.c_str() + "/clang-scan-deps"s;
  return CppToolchain(cc, ar, deps_scanner);
}
} // namespace pscm::build