//
// Created by PikachuHy on 2023/7/22.
//
module;
#include <pscm/common_def.h>
export module pscm.build:Artifact;
import std;
import pscm;
import :CompilationContext;
import :LinkingContext;
import :Label;

export namespace pscm::build {
class CppLibraryRule;

class Artifact {
public:
  [[nodiscard]] virtual bool is_cpp_library_artifact() const {
    return false;
  }
};

class CppLibraryArtifact : public Artifact {
public:
  [[nodiscard]] virtual bool is_cpp_library_artifact() const override {
    return true;
  }

  const CompilationContext& compilation_context() const {
    return compilation_context_;
  }

  const LinkingContext& linking_context() const {
    return linking_context_;
  }

private:
  std::vector<Label> deps_;
  std::vector<std::string> libs_;
  CompilationContext compilation_context_;
  LinkingContext linking_context_;
  friend class CppLibraryRule;
};

class CppBinaryArtifact : public Artifact {
public:
private:
};

} // namespace pscm::build