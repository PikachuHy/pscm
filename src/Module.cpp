#include "pscm/Module.h"
#include "pscm/ApiManager.h"
#include "pscm/Macro.h"
#include "pscm/Procedure.h"
#include "pscm/SchemeProxy.h"
#include "pscm/Str.h"
#include "pscm/SymbolTable.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"
#include <filesystem>
namespace fs = std::filesystem;

namespace pscm {

PSCM_DEFINE_BUILTIN_PROC(Module, "module-name") {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_module());
  auto m = arg.to_module();
  return m->name();
}

PSCM_DEFINE_BUILTIN_MACRO_PROC_WRAPPER(Module, "current-module", Label::APPLY_CURRENT_MODULE, "()") {
  return scm.current_module();
}

PSCM_DEFINE_BUILTIN_MACRO_PROC_WRAPPER(Module, "set-current-module", Label::APPLY_CURRENT_MODULE, "(module)") {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_sym());
  auto m = env->get(arg.to_symbol());
  PSCM_ASSERT(m.is_module());
  scm.set_current_module(m.to_module());
  return Cell::none();
}

std::string check_module(Cell name, const std::vector<std::string>& load_path_vec) {
  std::string path;
  for_each(
      [&path](Cell expr, auto) {
        PSCM_ASSERT(expr.is_sym());
        auto name = expr.to_symbol()->name();
        path += name;
        path += '/';
      },
      name);
  if (path.empty()) {
    PSCM_THROW_EXCEPTION("bad module name: " + name.to_string());
  }
  path.resize(path.size() - 1);
  path += ".scm";
  std::string fullname;
  bool module_found = false;
  for (int i = 0; i < load_path_vec.size(); ++i) {
    auto load_path = load_path_vec.at(i);
    if (!load_path.ends_with('/')) {
      load_path += '/';
    }
    fullname = load_path + path;
    if (fs::exists(fullname)) {
      module_found = true;
      break;
    }
  }
  if (!module_found) {
    SPDLOG_ERROR("file not exist: {}", fullname);
    PSCM_THROW_EXCEPTION("module not found: " + name.to_string());
  }
  return fullname;
}

std::vector<std::string> get_load_path(SymbolTable *env) {
  auto load_path = "%load-path"_sym;
  auto load_path_list = env->get_or(&load_path, nil);
  std::vector<std::string> load_path_vec;
  for_each(
      [&load_path_vec](Cell expr, auto) {
        PSCM_ASSERT(expr.is_str());
        load_path_vec.push_back(expr.to_str()->str());
      },
      load_path_list);
  if (load_path_vec.empty()) {
    load_path_vec.push_back(".");
  }
  auto c_env_load_path = getenv("PSCM_LOAD_PATH");
  std::string env_load_path;
  if (c_env_load_path) {
    env_load_path = c_env_load_path;
    SPDLOG_INFO("PSCM_LOAD_PATH: {}", env_load_path);
    load_path_vec.push_back(env_load_path);
  }
  return load_path_vec;
}

PSCM_DEFINE_BUILTIN_MACRO(Module, "resolve-module", Label::APPLY_RESOLVE_MODULE) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  auto name = scm.eval(env, arg);
  SPDLOG_INFO("module name: {}", name);
  PSCM_ASSERT(name.is_pair());
  auto load_path_vec = get_load_path(env);
  auto path = check_module(name, load_path_vec);
  SPDLOG_INFO("resolve module: {} from {}", name, path);
  if (!scm.has_module(name)) {
    scm.load_module(path, name);
    if (!scm.has_module(name)) {
      PSCM_THROW_EXCEPTION("load module " + name.to_string() + " error");
    }
  }
  auto m = scm.get_module(name);
  return m;
}

PSCM_DEFINE_BUILTIN_MACRO(Module, "use-modules", Label::APPLY_USE_MODULES) {
  PSCM_ASSERT(args.is_pair());
  auto load_path_vec = get_load_path(env);
  for_each(
      [&scm, &load_path_vec](Cell expr, auto) {
        PSCM_ASSERT(expr.is_pair());
        // TODO: use module
        auto path = check_module(expr, load_path_vec);
        SPDLOG_INFO("use module: {} from {}", expr, path);
        if (!scm.has_module(expr)) {
          scm.load_module(path, expr);
          if (!scm.has_module(expr)) {
            PSCM_THROW_EXCEPTION("load module " + expr.to_string() + " error");
          }
        }
        auto m = scm.get_module(expr);
        if (m == scm.current_module()) {
          PSCM_THROW_EXCEPTION("module cycle detected: " + Cell(m).to_string());
        }
        scm.current_module()->use_module(m);
      },
      args);
  return Cell::none();
}

PSCM_DEFINE_BUILTIN_MACRO_PROC_WRAPPER(Module, "module-use!", Label::TODO, "(module interface)") {
  PSCM_ASSERT(args.is_pair());
  auto arg1 = car(args);
  auto arg2 = cadr(args);
  PSCM_ASSERT(arg1.is_sym());
  PSCM_ASSERT(arg2.is_sym());
  arg1 = env->get(arg1.to_symbol());
  arg2 = env->get(arg2.to_symbol());
  PSCM_ASSERT(arg1.is_module());
  PSCM_ASSERT(arg2.is_module());
  auto m = arg1.to_module();
  auto m2 = arg2.to_module();
  m->use_module(m2, true);
  return Cell::none();
}

PSCM_DEFINE_BUILTIN_MACRO(Module, "export", Label::APPLY_EXPORT) {
  PSCM_ASSERT(args.is_pair());
  for_each(
      [&scm](Cell expr, auto) {
        PSCM_ASSERT(expr.is_sym());
        auto sym = expr.to_symbol();
        auto m = scm.current_module();
        m->export_symbol(sym);
      },
      args);
  return Cell::none();
}

std::ostream& operator<<(std::ostream& os, const Module& m) {
  os << '#';
  os << '<';
  os << "module";
  os << ' ';
  if (!m.name_.is_none()) {
    os << m.name_;
  }
  os << ' ';
  os << &m;
  os << '>';
  return os;
}

void Module::export_symbol(Symbol *sym) {
  if (export_sym_list_.contains(sym)) {
    PSCM_THROW_EXCEPTION(Cell(sym).to_string() + "has already exported");
  }
  export_sym_list_.insert(sym);
  SPDLOG_INFO("export symbol: {}", sym->name());
}

void Module::use_module(Module *m, bool use_all) {
  if (use_all) {
    PSCM_ASSERT(m->env());
    this->env_->use(*m->env());
  }
  else {
    for (auto sym : m->export_sym_list_) {
      PSCM_ASSERT(sym.is_sym());
      if (!m->env_->contains(sym.to_symbol())) {
        PSCM_THROW_EXCEPTION(sym.to_string() + " export from module " + m->name_.to_string() + ", but not found");
      }
      this->env_->use(m->env_, sym.to_symbol());
    }
  }
}

Cell Module::export_sym_list() {
  auto ret = cons(nil, nil);
  auto it = ret;
  for (auto key : export_sym_list_) {
    auto new_pair = cons(list(key, env_->get(key.to_symbol())), nil);
    it->second = new_pair;
    it = new_pair;
  }
  return ret->second;
}

PSCM_DEFINE_BUILTIN_PROC(Module, "module-ref") {
  PSCM_ASSERT(args.is_pair());
  auto module = car(args);
  auto name = cadr(args);
  PSCM_ASSERT(module.is_module());
  PSCM_ASSERT(name.is_sym());
  auto m = module.to_module();
  auto sym = name.to_symbol();
  if (*sym == "%module-public-interface"_sym) {
    auto sym_list = m->export_sym_list();
    return m->export_sym_list();
  }
  PSCM_THROW_EXCEPTION("not suppoted now");
}

PSCM_DEFINE_BUILTIN_MACRO_PROC_WRAPPER(Module, "module-map", Label::APPLY_APPLY, "(proc module)") {
  PSCM_ASSERT(args.is_pair());
  auto arg1 = car(args);
  auto arg2 = cadr(args);
  PSCM_ASSERT(arg1.is_sym());
  PSCM_ASSERT(arg2.is_sym());
  arg1 = env->get(arg1.to_symbol());
  arg2 = env->get(arg2.to_symbol());
  SPDLOG_INFO("args: {}, {}", arg1.to_string(), arg2.to_string());
  PSCM_ASSERT(arg1.is_proc());
  auto proc = arg1.to_proc();
  auto ret = cons(nil, nil);
  auto it = ret;
  while (arg2.is_pair()) {
    auto item = car(arg2);
    auto sym = car(item);
    auto val = cadr(item);
    Cell expr = list(arg1, list(quote, sym), list(quote, val));
    auto proc_ret = scm.eval(env, expr);
    auto new_pair = cons(proc_ret, nil);
    it->second = new_pair;
    it = new_pair;
    arg2 = cdr(arg2);
  }
  return list(quote, ret->second);
}

PSCM_DEFINE_BUILTIN_PROC(Module, "re-export") {
  return Cell::none();
}

PSCM_DEFINE_BUILTIN_MACRO(Module, "define-module", Label::APPLY_DEFINE_MODULE) {
  PSCM_ASSERT(args.is_pair());
  auto module_name = car(args);
  scm.create_module(module_name);
  return Cell::none();
}
} // namespace pscm