#include "Scheme.h"
#include "Evaluator.h"
#include "JIT.h"
#include "Parser.h"
#include "Procedure.h"
#include "Value.h"
#include "pscm/logger/Appender.h"
#include <fstream>
#include <iterator>
#include <llvm/Transforms/Utils/Cloning.h>
#include <pscm/common_def.h>
#include <string>

namespace pscm::core {
static llvm::ExitOnError exit_on_err;
PSCM_INLINE_LOG_DECLARE("pscm.core.Scheme");

class SchemeImpl {
public:
  SchemeImpl() {
    init();
  }

  void init() {
    if (has_init_) {
      return;
    }
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeAllAsmParsers();
    pscm::logger::Logger::root_logger()->add_appender(new pscm::logger::ConsoleAppender());
    has_init_ = true;
  }

  Value *run_jitted_func(AST *ast, std::unique_ptr<llvm::Module> module, bool print) {
    auto jit = exit_on_err(JIT::create());
    auto rt = jit->main_jit_dylib().createResourceTracker();
    auto tsm = llvm::orc::ThreadSafeModule(std::move(module), std::make_unique<llvm::LLVMContext>());
    exit_on_err(jit->add_module(std::move(tsm), rt));
    auto expr_sym = exit_on_err(jit->lookup("_anon_expr"));
    Value *ret = nullptr;
    if (auto map_expr = dynamic_cast<MapExprAST *>(ast); map_expr) {
      // Array
      Array *(*fp)() = expr_sym.getAddress().toPtr<Array *(*)()>();
      auto eval_ret = fp();
      std::vector<Value *> list;
      list.reserve(eval_ret->size);
      for (int i = 0; i < eval_ret->size; ++i) {
        list.push_back(new IntegerValue(eval_ret->data[i]));
      }
      ret = new ListValue(list);
      //      return new IntegerValue(eval_ret->size);
    }
    else if (auto call_expr = dynamic_cast<CallExprAST *>(ast); call_expr) {
      if (call_expr->type()) {
        if (auto array_type = dynamic_cast<const ArrayType *>(call_expr->type()); array_type) {
          Array *(*fp)() = expr_sym.getAddress().toPtr<Array *(*)()>();
          auto eval_ret = fp();
          std::vector<Value *> list;
          list.reserve(eval_ret->size);
          for (int i = 0; i < eval_ret->size; ++i) {
            list.push_back(new IntegerValue(eval_ret->data[i]));
          }
          ret = new ListValue(list);
        }
        else if (auto integer_type = dynamic_cast<const IntegerType *>(call_expr->type()); integer_type) {
          int (*fp)() = expr_sym.getAddress().toPtr<int (*)()>();
          auto eval_ret = fp();
          ret = new IntegerValue(eval_ret);
        }
        else {
          PSCM_UNIMPLEMENTED();
        }
      }
      else {
        PSCM_UNIMPLEMENTED();
      }
    }
    else {
      int (*fp)() = expr_sym.getAddress().toPtr<int (*)()>();
      auto eval_ret = fp();
      ret = new IntegerValue(eval_ret);
    }

    if (print) {
      assert(ret);
      std::cout << ret->to_string() << std::endl;
    }
    return ret;
  }

  Value *eval(const std::string& code, bool print) {
    auto llvm_ctx = std::make_unique<llvm::LLVMContext>();
    auto llvm_module = std::make_unique<llvm::Module>("pscm jit", *llvm_ctx);
    auto builder = std::make_unique<llvm::IRBuilder<>>(*llvm_ctx);

    Parser parser(code);
    Evaluator evaluator;
    CodegenContext ctx{
      .llvm_ctx = *llvm_ctx, .llvm_module = *llvm_module, .builder = *builder, .evaluator = evaluator
    };
    llvm_module->getOrInsertFunction("car_array[integer]",
                                     llvm::FunctionType::get(llvm::Type::getInt64Ty(*llvm_ctx),
                                                             { llvm::PointerType::get(ctx.get_array(), 0) }, false));
    auto value = parser.parse_one();
    AST *ast = nullptr;
    Value *ret = nullptr;
    while (value) {
      if (auto proc = dynamic_cast<Procedure *>(value); proc) {
        evaluator.add_proc(proc->name(), proc);
        ctx.proc_map[proc->name()->to_string()] = proc;
        value = parser.parse_one();
        continue;
      }
      ast = evaluator.eval(value);
      if (auto expr = dynamic_cast<ExprAST *>(ast); expr) {

        auto proto = new PrototypeAST("__anon_expr", {}, {});
        auto func = new FunctionAST(proto, ast);

        [[maybe_unused]] auto _ = func->codegen(ctx);
        // llvm::errs() << ctx.llvm_module;
        // llvm::errs() << "\n";
        llvm::verifyModule(ctx.llvm_module);
        run_jitted_func(ast, llvm::CloneModule(ctx.llvm_module), print);
        ctx.llvm_module.getFunction("__anon_expr")->eraseFromParent();
        value = parser.parse_one();
      }
      else {
        PSCM_ASSERT(ast);
        [[maybe_unused]] auto v = ast->codegen(ctx);
        value = parser.parse_one();
      }
    }
    return ret;
  }

  bool has_init_ = false;
};

Scheme::Scheme()
    : impl_(new SchemeImpl()) {
}

bool Scheme::load(const std::string& filename, bool print) {
  std::ifstream file(filename);
  std::string content;

  if (file) {
    content.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    file.close();
  }
  auto ret = impl_->eval(content, print);
  return ret;
}

Value *Scheme::eval(const char *code) {
  return impl_->eval(code, /*print=*/false);
}

Scheme::~Scheme() {
  delete impl_;
}

void Scheme::set_logger_level(int level) {
  pscm::logger::Logger::get_logger("pscm")->set_level(static_cast<logger::Level>(level));
}
} // namespace pscm::core
