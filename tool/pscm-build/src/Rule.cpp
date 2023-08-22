//
// Created by PikachuHy on 2023/7/22.
//
module;
#include <pscm/common_def.h>
module pscm.build;
import pscm;
import std;
import fmt;
import glob;
import subprocess;
import :Rule;
import :Artifact;
import :DepsScanner;
import :BuildVariables;
import :Action;
import :CompilationContext;
import :LinkingContext;
import :Label;
import :RuleContext;

namespace pscm::build {

void make_sure_parent_path_exist(const std::string& filename) {
  auto path = fs::path(filename).parent_path();
  if (!fs::exists(path)) {
    PSCM_INFO("create directory: {}", path);
    fs::create_directories(path);
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

void parse_label(RuleContext ctx, Cell args, std::vector<Label>& labels) {
  for_each(
      [&ctx, &labels](Cell expr, auto) {
        if (expr.is_str()) {
          auto s = expr.to_str()->str();
          auto label = Label::parse(s);
          if (label.has_value()) {
            if (label->package().empty()) {
              label = Label(label->repo(), ctx.package(), label->name());
            }
            labels.push_back(label.value());
          }
          else {
            PSCM_THROW_EXCEPTION("bad label: " + s);
          }
        }
        else {
          PSCM_THROW_EXCEPTION("bad expr: " + expr.to_string());
        }
      },
      args);
}

std::string get_relative_path(std::string_view package, std::string_view filename) {
  if (package.empty()) {
    return std::string(filename);
  }
  return fmt::format("{}/{}", package.substr(1), filename);
}

void parse_file(RuleContext ctx, Cell args, std::vector<std::string>& files) {
  while (args.is_pair()) {
    auto arg = car(args);
    if (arg.is_str()) {
      files.push_back(get_relative_path(ctx.package(), arg.to_str()->str()));
    }
    else if (arg.is_pair()) {
      if (car(arg).is_sym() && car(arg) == "glob"_sym) {
        auto glob_args = cdr(arg);
        while (glob_args.is_pair()) {
          auto s = car(glob_args);
          PSCM_ASSERT(s.is_str());
          std::string file_regex = std::string(s.to_str()->str());
          PSCM_DEBUG("glob: {}", file_regex);
          for (auto& p : glob::rglob(file_regex)) {
            files.push_back(get_relative_path(ctx.package(), p.string()));
          }
          glob_args = cdr(glob_args);
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

struct Vertex;

struct Graph {
  std::vector<Vertex *> topological_sort(const std::vector<Vertex *>& vertex_list);
};

struct Vertex {
  CompileBuildVariables *data;
  std::unordered_set<Vertex *> neighbor;
};

std::vector<Vertex *> Graph::topological_sort(const std::vector<Vertex *>& vertex_list) {
  std::unordered_set<Vertex *> vertex_set(vertex_list.begin(), vertex_list.end());
  std::vector<Vertex *> ret;
  std::queue<Vertex *> q;
  for (auto vertex : vertex_set) {
    if (vertex->neighbor.empty()) {
      q.push(vertex);
    }
  }
  while (!q.empty()) {
    auto vertex0 = q.front();
    q.pop();
    ret.push_back(vertex0);
    for (auto vertex : vertex_set) {
      auto it = vertex->neighbor.find(vertex0);
      if (it != vertex->neighbor.end()) {
        it = vertex->neighbor.erase(it);
        if (vertex->neighbor.empty()) {
          q.push(vertex);
        }
      }
    }
  }
  return ret;
}

class CppHelper {
public:
  std::string get_obj_path(std::string source_path) {
    return fmt::format("pscm-build-bin{}/_objs/{}.o", package_, source_path);
  }

  std::string get_bmi_path(std::string source_path) {
    return fmt::format("pscm-build-bin{}/_bmis/{}", package_, source_path);
  }

  auto get_module_bmi_path(const std::string& name, auto& module_bmi_map, auto& module_name_map) -> std::string {
    auto it = module_bmi_map.find(name);
    if (it == module_bmi_map.end()) {
      auto it2 = module_name_map.find(name);
      if (it2 == module_name_map.end()) {
        PSCM_ERROR("can not find module: {}", name);
        std::exit(1);
      }
      else {
        return it2->second->bmi;
      }
    }
    else {
      return it->second;
    }
  }

  auto compile_cpp_sources(const std::vector<Artifact *>& artifact_list) -> void {
    PSCM_ASSERT(rule_);
    auto module_name_map = ctx_->get_module_name_map();
    std::unordered_map<std::string, std::string> module_src_map;
    std::unordered_map<std::string, std::string> module_bmi_map;
    std::unordered_map<std::string, ModuleDep> dep_map;
    std::unordered_map<std::string, Vertex *> vertex_map;
    std::vector<Vertex *> vertex_list;
    for (const auto& src : srcs_) {
      std::vector<std::string> args;
      args.push_back(src);
      std::string object_file = get_obj_path(src);
      args.push_back(object_file);
      object_files.push_back(object_file);
      auto variables = new CompileBuildVariables();
      variables->set_source_file(src);
      variables->set_output_file(object_file);
      variables->set_opts(copts_);
      variables->set_includes(includes_);
      variables->set_defines(defines_);
      auto depsScanner = new DepsScanner(ctx_);
      auto dep = depsScanner->scan(*variables, repo_path_);
      PSCM_TRACE("requires: {}", dep.require_modules());
      PSCM_TRACE("provide: {}", dep.provide_module());
      if (dep.is_module_interface()) {
        auto module_name = dep.provide_module().value();
        module_src_map[module_name] = src;
        std::string module_output_path;
        std::string original_module_name = module_name;
        std::string valid_basename;
        auto index = module_name.find_last_of(":");
        if (index != std::string::npos) {
          valid_basename = module_name.replace(index, 1, "-");
        }
        else {
          valid_basename = module_name;
        }
        std::string basename;
        auto slash_index = src.find_last_of('/');
        if (slash_index == std::string::npos) {
          basename = "";
        }
        else {
          basename = src.substr(0, slash_index);
        }
        if (basename.empty()) {
          module_output_path = valid_basename + ".pcm";
        }
        else {
          module_output_path = basename + '/' + valid_basename + ".pcm";
        }
        module_output_path = get_bmi_path(module_output_path);
        module_bmi_map[original_module_name] = module_output_path;
        PSCM_INFO("add module {} --> {}", original_module_name, module_output_path);
        modules_.push_back(new ModuleInfo{ .name = original_module_name, .bmi = module_output_path });
      }
      dep_map[src] = dep;
      auto vertex = new Vertex();
      vertex->data = variables;
      vertex_map[src] = vertex;
      vertex_list.push_back(vertex);
    }
    // create DAG
    for (const auto& src : srcs_) {
      auto dep = dep_map[src];
      auto vertex = vertex_map[src];
      for (const auto& name : dep.require_modules()) {
        auto it = module_src_map.find(name);
        if (it == module_src_map.end()) {
          auto it2 = module_name_map.find(name);
          if (it2 == module_name_map.end()) {
            PSCM_ERROR("module {} not found", name);
            std::exit(1);
          }
          else {
            // do nothing now
          }
        }
        else {
          auto module_src = it->second;
          auto module_vertex = vertex_map.at(module_src);
          vertex->neighbor.insert(module_vertex);
        }
      }
    }
    auto new_vertex_list = Graph().topological_sort(vertex_list);
    for (auto vertex : new_vertex_list) {
      auto variables = vertex->data;
      auto src = variables->get_source_file();
      auto object_file = variables->get_output_file();
      auto dep = dep_map[src];
      std::vector<ModuleInfo> modules;
      auto v = new CompileBuildVariables(*variables);
      for (const auto& name : dep.require_modules()) {
        std::string bmi_path = get_module_bmi_path(name, module_bmi_map, module_name_map);
        modules.emplace_back(name, bmi_path);
      }
      v->set_modules(modules);
      if (dep.is_module_interface()) {
        auto cur_module_name = dep.provide_module().value();
        auto cur_it = module_bmi_map.find(cur_module_name);
        if (cur_it == module_bmi_map.end()) {
          PSCM_ERROR("can not find module: {}", cur_module_name);
          std::exit(1);
        }
        auto module_output_path = cur_it->second;
        v->set_output_file(module_output_path);
        CppModuleInterfaceCompileAction(ctx_, *v, &CppRuleBase::toolchain()).run(repo_path_);
        v->set_source_file(module_output_path);
        v->set_output_file(object_file);
        CppModuleInterfaceCodegenAction(ctx_, *v, &CppRuleBase::toolchain()).run(repo_path_);
      }
      else {
        v->set_output_file(object_file);
        CppModuleImplementationCompileActon(ctx_, *v, &CppRuleBase::toolchain()).run(repo_path_);
      }
    }
  }

  std::vector<std::string> srcs_;
  std::vector<std::string> copts_;
  std::vector<std::string> includes_;
  std::vector<std::string> defines_;
  CompilationContext *ctx_;
  CppRuleBase *rule_;
  std::vector<std::string> object_files;
  std::vector<ModuleInfo *> modules_;
  std::string repo_path_;
  std::string package_;
};

void CppRuleBase::parse(RuleContext ctx, Cell args) {
  while (args.is_pair()) {
    auto arg = car(args);
    if (!arg.is_pair()) {
      PSCM_THROW_EXCEPTION("bad expr: " + args.to_string());
    }
    parse_attr(ctx, arg);
    args = cdr(args);
  }
}

void CppLibraryRule::parse_attr(RuleContext ctx, Cell args) {
  auto arg = car(args);
  if (!arg.is_sym()) {
    PSCM_THROW_EXCEPTION("bad expr: " + args.to_string());
  }
  if (arg == "hdrs"_sym) {
    auto hdrs = cdr(args);
    parse_file(ctx, hdrs, hdrs_);
  }
  else if (arg == "defines"_sym) {
    parse_string(cdr(args), this->defines_);
  }
  else if (arg == "includes"_sym) {
    parse_string(cdr(args), this->includes_);
  }
  else {
    CppRuleBase::parse_attr(ctx, args);
  }
}

void CppRuleBase::parse_attr(RuleContext ctx, Cell args) {
  auto arg = car(args);
  if (!arg.is_sym()) {
    PSCM_THROW_EXCEPTION("bad expr: " + args.to_string());
  }
  if (arg == "name"_sym) {
    auto s = cadr(args);
    PSCM_ASSERT(s.is_str());
    name_ = s.to_str()->str();
    label_ = Label(ctx.repo(), ctx.package(), name_);
  }
  else if (arg == "srcs"_sym) {
    auto srcs = cdr(args);
    parse_file(ctx, srcs, srcs_);
  }
  else if (arg == "copts"_sym) {
    auto copts = cdr(args);
    parse_string(copts, this->copts_);
  }
  else if (arg == "deps"_sym) {
    parse_label(ctx, cdr(args), this->deps_);
  }
  else {
    PSCM_THROW_EXCEPTION("unknown type: " + arg.to_string());
  }
}

void CppBinaryRule::parse_attr(RuleContext ctx, Cell args) {
  CppRuleBase::parse_attr(ctx, args);
}

Artifact *CppLibraryRule::run(std::string_view repo_path, std::string_view package,
                              const std::unordered_set<Artifact *>& depset) {
  auto compilation_context = init_compilation_context(depset);
  auto linking_context = init_linking_context(depset);
  PSCM_INFO("Compiling {}", name());
  std::vector<Artifact *> artifact_list;
  CppHelper cpp_helper{};
  cpp_helper.srcs_ = srcs_;
  cpp_helper.copts_ = copts_;
  cpp_helper.includes_ = includes_;
  cpp_helper.defines_ = defines_;
  cpp_helper.rule_ = this;
  cpp_helper.ctx_ = &compilation_context;
  cpp_helper.repo_path_ = repo_path;
  cpp_helper.package_ = package;
  cpp_helper.compile_cpp_sources(artifact_list);
  auto object_files = cpp_helper.object_files;
  std::unordered_set<std::string> libs;
  if (!srcs_.empty()) {
    PSCM_INFO("Linking {}", name());
    std::string libname = "pscm-build-bin/lib" + name_ + ".a";
    LinkBuildVariables var;
    var.set_output_file(libname);
    var.set_object_files(object_files);
    CppLinkStaticAction(&linking_context, var, &toolchain()).run(repo_path);
    LinkingContext::Builder builder;
    builder.add_library(libname);
    linking_context.merge(builder.build());
  }
  {
    CompilationContext::Builder builder;
    builder.add_includes(includes_);
    builder.add_defines(defines_);
    builder.add_modules(cpp_helper.modules_);
    compilation_context.merge(builder.build());
  }
  auto artifact = new CppLibraryArtifact();
  artifact->compilation_context_ = compilation_context;
  artifact->linking_context_ = linking_context;
  artifact->deps_ = deps_;
  artifact->libs_ = std::vector<std::string>(libs.begin(), libs.end());
  return artifact;
}

Artifact *CppBinaryRule::run(std::string_view repo_path, std::string_view package,
                             const std::unordered_set<Artifact *>& depset) {
  auto compilation_context = init_compilation_context(depset);
  auto linking_context = init_linking_context(depset);
  PSCM_INFO("Compiling {}", name());
  std::vector<Artifact *> artifact_list;
  CppHelper cpp_helper{};
  cpp_helper.srcs_ = srcs_;
  cpp_helper.copts_ = copts_;
  cpp_helper.rule_ = this;
  cpp_helper.ctx_ = &compilation_context;
  cpp_helper.repo_path_ = repo_path;
  cpp_helper.package_ = package;
  cpp_helper.compile_cpp_sources(artifact_list);
  auto object_files = cpp_helper.object_files;
  PSCM_INFO("Linking {}", name());
  LinkBuildVariables var;
  var.set_output_file("pscm-build-bin/" + name_);
  var.set_object_files(object_files);
  CppLinkBinaryAction(&linking_context, var, &toolchain()).run(repo_path);
  auto artifact = new CppBinaryArtifact();

  return artifact;
}

void RuleRunner::run(Label label) {
  if (label.name() == "...") {
    PSCM_ERROR("build ... not supported now");
    std::exit(1);
  }
  else if (label.name() == "all") {
    for (auto& entry : rule_map_) {
      run_rule(entry.second, label.package());
    }
  }
  else {
    auto it = rule_map_.find(label);
    if (it == rule_map_.end()) {
      PSCM_THROW_EXCEPTION(fmt::format("target not found: {}", label.name()));
    }
    auto rule = it->second;
    run_rule(rule, label.package());
  }
}

void RuleRunner::run_rule(Rule *rule, std::string_view package) {
  if (artifact_map_.find(rule->label()) != artifact_map_.end()) {
    return;
  }
  for (const auto& dep : rule->deps()) {
    auto it = rule_map_.find(dep);
    if (it == rule_map_.end()) {
      PSCM_THROW_EXCEPTION(rule->label().to_string() + " dep not found: " + dep.to_string());
    }
    auto r = it->second;
    run_rule(r, package);
  }
  std::unordered_set<Artifact *> depset;
  collect_dep(depset, rule);
  auto artifact = rule->run(repo_path_, package, depset);
  artifact_map_[rule->label()] = artifact;
}
} // namespace pscm::build