#include <fstream>
#include <glob/glob.h>
#include <iostream>
#include <pscm/Parser.h>
#include <pscm/Scheme.h>
#include <pscm/Str.h>
#include <pscm/common_def.h>
#include <pscm/scm_utils.h>
#include <regex>
#include <spdlog/spdlog.h>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#if PSCM_STD_COMPAT
#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif
using namespace pscm;
class CppLibraryRule;

class Artifact {
public:
  virtual std::vector<std::string> get(StringView key) const = 0;
  ;
};

class CppLibraryArtifact : public Artifact {
public:
  std::vector<std::string> get(StringView key) const override {
    if (key == "defines") {
      return defines_;
    }
    else if (key == "includes") {
      return includes_;
    }
    else if (key == "deps") {
      return deps_;
    }
    else if (key == "library") {
      return libs_;
    }
    else {
      PSCM_THROW_EXCEPTION("bad key: " + std::string(key));
    }
  }

private:
  std::vector<std::string> defines_;
  std::vector<std::string> includes_;
  std::vector<std::string> deps_;
  std::vector<std::string> libs_;
  friend class CppLibraryRule;
};

class CppBinaryArtifact : public Artifact {
public:
  std::vector<std::string> get(StringView key) const override {
    PSCM_THROW_EXCEPTION("bad key: " + std::string(key));
  }

private:
};

class Rule {
public:
  virtual void parse(Cell args) = 0;
  virtual Artifact *run(const std::unordered_set<Artifact *>& depset) = 0;
  virtual const std::string& name() const = 0;
  virtual const std::vector<std::string>& deps() const = 0;
};

class CppLibraryRule : public Rule {
public:
  void parse(Cell args) override {
    while (args.is_pair()) {
      auto arg = car(args);
      if (!arg.is_pair()) {
        PSCM_THROW_EXCEPTION("bad expr: " + args.to_string());
      }
      parse_attr(arg);
      args = cdr(args);
    }
  }

  Artifact *run(const std::unordered_set<Artifact *>& depset) override {
    std::vector<std::string> object_files;
    SPDLOG_INFO("Compiling");
    std::vector<Artifact *> artifact_list;
    for (const auto& dep : depset) {
      artifact_list.push_back(dep);
    }
    for (const auto& src : srcs_) {
      std::string object_file = src + ".o";
      object_files.push_back(object_file);
      auto command_line = construct_compile_command_line(src, object_file, artifact_list);
      SPDLOG_INFO("RUN: {}", command_line);
      system(command_line.c_str());
    }
    std::unordered_set<std::string> libs;
    if (!srcs_.empty()) {
      SPDLOG_INFO("Linking");
      std::string libname = "lib" + name_ + ".a";
      auto command_line = construct_link_command_line(libname, object_files, artifact_list);
      SPDLOG_INFO("RUN: {}", command_line);
      system(command_line.c_str());
      libs.insert(libname);
    }
    auto artifact = new CppLibraryArtifact();
    artifact->defines_ = defines_;
    artifact->includes_ = includes_;
    artifact->deps_ = deps_;
    for (const auto& dep : depset) {
      auto dep_libs = dep->get("library");
      libs.insert(dep_libs.begin(), dep_libs.end());
    }
    artifact->libs_ = std::vector<std::string>(libs.begin(), libs.end());
    return artifact;
  }

  const std::string& name() const override {
    return name_;
  }

  const std::vector<std::string>& deps() const override {
    return deps_;
  }

private:
  void parse_attr(Cell args) {
    auto arg = car(args);
    if (!arg.is_sym()) {
      PSCM_THROW_EXCEPTION("bad expr: " + args.to_string());
    }
    if (arg == "name"_sym) {
      auto s = cadr(args);
      PSCM_ASSERT(s.is_str());
      name_ = s.to_str()->str();
    }
    else if (arg == "srcs"_sym) {
      auto srcs = cdr(args);
      parse_file(srcs, srcs_);
    }
    else if (arg == "hdrs"_sym) {
      auto hdrs = cdr(args);
      parse_file(hdrs, hdrs_);
    }
    else if (arg == "copts"_sym) {
      auto copts = cdr(args);
      parse_string(copts, this->copts_);
    }
    else if (arg == "defines"_sym) {
      parse_string(cdr(args), this->defines_);
    }
    else if (arg == "includes"_sym) {
      parse_string(cdr(args), this->includes_);
    }
    else if (arg == "deps"_sym) {
      parse_string(cdr(args), this->deps_);
    }
    else {
      PSCM_THROW_EXCEPTION("unknown type: " + arg.to_string());
    }
  }

  void parse_file(Cell args, std::vector<std::string>& files) {
    while (args.is_pair()) {
      auto arg = car(args);
      if (arg.is_str()) {
        files.push_back(std::string(arg.to_str()->str()));
      }
      else if (arg.is_pair()) {
        if (car(arg).is_sym() && car(arg) == "glob"_sym) {
          auto s = cadr(arg);
          PSCM_ASSERT(s.is_str());
          std::string file_regex = std::string(s.to_str()->str());
          std::cout << "glob: " << file_regex << std::endl;
          std::regex re(file_regex, std::regex_constants::egrep);
          for (auto& p : glob::glob(file_regex)) {
            std::cout << p << std::endl;
            files.push_back(p.string());
          }
        }
        else {
          PSCM_THROW_EXCEPTION("unknown tag: " + car(arg).to_string());
        }
      }
      else {
        PSCM_THROW_EXCEPTION("bad file: " + arg.to_string());
      }
      args = cdr(args);
    }
  }

  void parse_string(Cell args, std::vector<std::string>& list) {
    for_each(
        [&list](Cell expr, auto) {
          if (expr.is_str()) {
            auto s = expr.to_str()->str();
            list.push_back(std::string(s));
          }
          else {
            PSCM_THROW_EXCEPTION("bad expr: " + expr.to_string());
          }
        },
        args);
  }

  std::string construct_compile_command_line(StringView input, StringView output,
                                             const std::vector<Artifact *>& artifact_list) {
    std::stringstream ss;
    ss << "clang++";
    ss << " ";

    for (const auto& arg : defines_) {
      ss << "-D";
      ss << arg;
      ss << ' ';
    }
    for (const auto& arg : includes_) {
      ss << "-I";
      ss << ' ';
      ss << arg;
      ss << ' ';
    }
    for (const auto& arg : copts_) {
      ss << arg;
      ss << ' ';
    }
    for (const auto& artifact : artifact_list) {
      auto defines = artifact->get("defines");

      for (const auto& arg : defines) {
        ss << "-D";
        ss << arg;
        ss << ' ';
      }

      auto includes = artifact->get("includes");

      for (const auto& arg : includes) {
        ss << "-I";
        ss << ' ';
        ss << arg;
        ss << ' ';
      }
    }
    ss << "-c";
    ss << ' ';

    ss << input;
    ss << ' ';

    ss << "-o";
    ss << ' ';
    ss << output;
    return ss.str();
  }

  std::string construct_link_command_line(StringView libname, const std::vector<std::string>& object_files,
                                          const std::vector<Artifact *>& artifact_list) {
    std::stringstream ss;
    ss << "llvm-ar";
    ss << ' ';

    ss << "-rcs";
    ss << ' ';

    ss << libname;
    ss << ' ';

    for (const auto& object_file : object_files) {
      ss << object_file;
      ss << ' ';
    }
    return ss.str();
  }

private:
  std::string name_;
  std::vector<std::string> srcs_;
  std::vector<std::string> hdrs_;
  std::vector<std::string> copts_;
  std::vector<std::string> defines_;
  std::vector<std::string> includes_;
  std::vector<std::string> deps_;
};

class CppBinaryRule : public Rule {
public:
  void parse(Cell args) override {
    while (args.is_pair()) {
      auto arg = car(args);
      if (!arg.is_pair()) {
        PSCM_THROW_EXCEPTION("bad expr: " + args.to_string());
      }
      parse_attr(arg);
      args = cdr(args);
    }
  }

  Artifact *run(const std::unordered_set<Artifact *>& depset) override {
    std::vector<std::string> object_files;
    SPDLOG_INFO("Compiling");
    std::vector<Artifact *> artifact_list;
    // TOOD: depset
    for (const auto& dep : depset) {
      artifact_list.push_back(dep);
    }
    for (const auto& src : srcs_) {
      std::vector<std::string> args;
      args.push_back(src);
      std::string object_file = src + ".o";
      args.push_back(object_file);
      object_files.push_back(object_file);
      auto command_line = construct_compile_command_line(src, object_file, artifact_list);
      SPDLOG_INFO("RUN: {}", command_line);
      system(command_line.c_str());
    }
    SPDLOG_INFO("Linking");
    auto command_line = construct_link_command_line(object_files, artifact_list);
    SPDLOG_INFO("RUN: {}", command_line);
    system(command_line.c_str());
    auto artifact = new CppBinaryArtifact();

    return artifact;
  }

  const std::string& name() const override {
    return name_;
  }

  const std::vector<std::string>& deps() const override {
    return deps_;
  }

private:
  void parse_attr(Cell args) {
    auto arg = car(args);
    if (!arg.is_sym()) {
      PSCM_THROW_EXCEPTION("bad expr: " + args.to_string());
    }
    if (arg == "name"_sym) {
      auto s = cadr(args);
      PSCM_ASSERT(s.is_str());
      name_ = s.to_str()->str();
    }
    else if (arg == "srcs"_sym) {
      auto srcs = cdr(args);
      parse_file(srcs, srcs_);
    }
    else if (arg == "copts"_sym) {
      auto copts = cdr(args);
      parse_string(copts, this->copts_);
    }
    else if (arg == "deps"_sym) {
      parse_string(cdr(args), this->deps_);
    }
    else {
      PSCM_THROW_EXCEPTION("unknown type: " + arg.to_string());
    }
  }

  void parse_file(Cell args, std::vector<std::string>& files) {
    while (args.is_pair()) {
      auto arg = car(args);
      if (arg.is_str()) {
        files.push_back(std::string(arg.to_str()->str()));
      }
      else if (arg.is_pair()) {
        if (car(arg).is_sym() && car(arg) == "glob"_sym) {
          auto s = cadr(arg);
          PSCM_ASSERT(s.is_str());
          std::string file_regex = std::string(s.to_str()->str());
          std::cout << "glob: " << file_regex << std::endl;
          std::regex re(file_regex, std::regex_constants::egrep);
          for (auto& p : glob::glob(file_regex)) {
            std::cout << p << std::endl;
            files.push_back(p.string());
          }
        }
        else {
          PSCM_THROW_EXCEPTION("unknown tag: " + car(arg).to_string());
        }
      }
      else {
        PSCM_THROW_EXCEPTION("bad file: " + arg.to_string());
      }
      args = cdr(args);
    }
  }

  void parse_string(Cell args, std::vector<std::string>& list) {
    for_each(
        [&list](Cell expr, auto) {
          if (expr.is_str()) {
            auto s = expr.to_str()->str();
            list.push_back(std::string(s));
          }
          else {
            PSCM_THROW_EXCEPTION("bad expr: " + expr.to_string());
          }
        },
        args);
  }

  std::string construct_compile_command_line(StringView input, StringView output,
                                             const std::vector<Artifact *>& artifact_list) {
    std::stringstream ss;
    ss << "clang++";
    ss << " ";
    for (const auto& arg : copts_) {
      ss << arg;
      ss << ' ';
    }
    for (const auto& artifact : artifact_list) {
      auto defines = artifact->get("defines");

      for (const auto& arg : defines) {
        ss << "-D";
        ss << arg;
        ss << ' ';
      }

      auto includes = artifact->get("includes");

      for (const auto& arg : includes) {
        ss << "-I";
        ss << ' ';
        ss << arg;
        ss << ' ';
      }
    }
    ss << "-c";
    ss << ' ';

    ss << input;
    ss << ' ';

    ss << "-o";
    ss << ' ';
    ss << output;
    return ss.str();
  }

  std::string construct_link_command_line(const std::vector<std::string>& object_files,
                                          const std::vector<Artifact *>& artifact_list) {
    std::stringstream ss;
    ss << "clang++";
    ss << ' ';

    ss << "-o";
    ss << ' ';
    ss << name_;
    ss << ' ';

    for (const auto& object_file : object_files) {
      ss << object_file;
      ss << ' ';
    }

    for (const auto& dep : artifact_list) {
      auto libs = dep->get("library");
      for (const auto& lib : libs) {
        ss << lib;
        ss << " ";
      }
    }
    return ss.str();
  }

private:
  std::string name_;
  std::vector<std::string> srcs_;
  std::vector<std::string> copts_;
  std::vector<std::string> deps_;
};

Rule *_cpp_library_impl(Cell args) {
  auto rule = new CppLibraryRule();
  rule->parse(args);
  return rule;
}

Rule *_cpp_binary_impl(Cell args) {
  auto rule = new CppBinaryRule();
  rule->parse(args);
  return rule;
}

class RuleRunner {
public:
  RuleRunner(std::unordered_map<std::string, Rule *> rule_map)
      : rule_map_(rule_map) {
  }

  void run(const std::string& target) {
    if (target == ":all") {
      for (auto& entry : rule_map_) {
        run_rule(entry.second);
      }
    }
    else {
      auto it = rule_map_.find(target);
      if (it == rule_map_.end()) {
        PSCM_THROW_EXCEPTION("target not found: " + target);
      }
      auto rule = it->second;
      run_rule(rule);
    }
  }

  void run_rule(Rule *rule) {
    if (artifact_map_.find(":" + rule->name()) != artifact_map_.end()) {
      return;
    }
    for (const auto& dep : rule->deps()) {
      auto it = rule_map_.find(dep);
      if (it == rule_map_.end()) {
        PSCM_THROW_EXCEPTION("dep not found: " + dep);
      }
      auto r = it->second;
      run_rule(r);
    }
    std::unordered_set<Artifact *> depset;
    collect_dep(depset, rule);
    std::cout << "depset size: " << depset.size() << std::endl;
    auto artifact = rule->run(depset);
    artifact_map_[":" + rule->name()] = artifact;
  }

  void collect_dep(std::unordered_set<Artifact *>& depset, Rule *rule) {
    for (const auto& dep : rule->deps()) {
      auto r = rule_map_.at(dep);
      collect_dep(depset, r);
      auto artifact = artifact_map_.at(dep);
      depset.insert(artifact);
    }
  }

private:
  std::unordered_map<std::string, Rule *> rule_map_;
  std::unordered_map<std::string, Artifact *> artifact_map_;
};

int main(int argc, char **argv) {
  std::string target = ":all";
  if (argc >= 2) {
    target = argv[1];
    if (target.front() != ':') {
      SPDLOG_ERROR("bad target: " + target);
      return -1;
    }
  }
  std::string filename = "build.pscm";
  bool ok = fs::exists(filename);
  if (!ok) {
    SPDLOG_ERROR("build.pscm is required!!!");
    return 1;
  }
  std::fstream ifs;
  ifs.open(filename, std::ios::in);
  if (!ifs.is_open()) {
    SPDLOG_ERROR("open {} failed", filename);
    return 2;
  }

  ifs.seekg(0, ifs.end);
  auto sz = ifs.tellg();
  ifs.seekg(0, ifs.beg);
  std::string code;
  code.resize(sz);
  ifs.read((char *)code.data(), sz);
  std::unordered_map<std::string, Rule *> rule_map;
  try {
    Parser parser(code, filename);
    Cell expr = parser.next();
    while (!expr.is_none()) {
      if (expr.is_pair() && car(expr).is_sym()) {
        auto rule = car(expr);
        if (rule == "cpp_library"_sym) {
          auto rule = _cpp_library_impl(cdr(expr));
          rule_map[":" + std::string(rule->name())] = rule;
        }
        else if (rule == "cpp_binary"_sym || rule == "cpp_test"_sym) {
          auto rule = _cpp_binary_impl(cdr(expr));
          rule_map[":" + rule->name()] = rule;
        }
        else {
          SPDLOG_INFO("rule {} not supported now", rule);
        }
      }
      expr = parser.next();
    }
    RuleRunner runner(rule_map);
    runner.run(target);
  }

  catch (Exception& ex) {
    SPDLOG_ERROR("load file {} error", filename);
  }
  return 0;
}
