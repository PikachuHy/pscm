#include "pscm.h"

#include "pscm/Parser.h"
#include "pscm/Str.h"
#include "pscm/Symbol.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"

#include "assert.h"

using namespace pscm;

SCM *translate(Cell ret) {
  SCM_DEBUG_TRANS("translate %s\n", ret.to_std_string().c_str());
  if (ret.is_none()) {
    return scm_none();
  }
  if (!ret.is_pair()) {
    if (ret.is_bool()) {
      if (ret.to_bool()) {
        return scm_bool_true();
      }
      else {
        return scm_bool_false();
      }
    }
    if (ret.is_sym()) {
      std::string sym_str;
      ret.to_sym()->to_string().toUTF8String(sym_str);
      auto sym = create_sym(sym_str.c_str(), sym_str.length());
      return sym;
    }
    if (ret.is_str()) {
      std::string sym_str;
      ret.to_str()->str().toUTF8String(sym_str);
      auto sym = create_sym(sym_str.c_str(), sym_str.length());
      sym->type = SCM::STR;
      return sym;
    }

    printf("%s:%d [%s] not supported %s\n", __BASE_FILE__, __LINE__, __func__, car(ret).to_std_string().c_str());
    std::exit(1);
  }
  SCM_List dummy;
  dummy.data = nullptr;
  dummy.data = nullptr;
  SCM_List *it = &dummy;
  if (ret.is_pair()) {
    while (ret.is_pair()) {
      auto first = car(ret);
      if (first.is_pair()) {
        SCM_List *pair = new SCM_List();
        pair->data = translate(first);
        // print_ast(pair->data);
        // printf("\n");
        pair->next = nullptr;
        it->next = pair;
        it = pair;
        ret = cdr(ret);
      }
      else if (first.is_sym()) {
        std::string sym_str;
        car(ret).to_sym()->to_string().toUTF8String(sym_str);
        auto sym = create_sym(sym_str.c_str(), sym_str.length());
        SCM_List *pair = new SCM_List();
        pair->data = sym;
        pair->next = nullptr;
        it->next = pair;
        it = pair;
        ret = cdr(ret);
      }
      else if (first.is_nil()) {
        SCM_List *pair = new SCM_List();
        pair->data = scm_nil();
        pair->next = nullptr;
        it->next = pair;
        it = pair;
        ret = cdr(ret);
        // break;
      }
      else if (first.is_num()) {
        SCM_List *pair = new SCM_List();
        SCM *data = new SCM();
        data->type = SCM::NUM;
        auto val = first.to_num()->to_int();
        data->value = (void *)val;
        pair->data = data;
        it->next = pair;
        it = pair;
        ret = cdr(ret);
      }
      else if (first.is_bool()) {
        SCM_List *pair = new SCM_List();
        pair->data = first.to_bool() ? scm_bool_true() : scm_bool_false();
        it->next = pair;
        it = pair;
        ret = cdr(ret);
      }
      else if (first.is_str()) {
        std::string sym_str;
        car(ret).to_str()->str().toUTF8String(sym_str);
        auto sym = create_sym(sym_str.c_str(), sym_str.length());
        // set type to String
        sym->type = SCM::STR;
        SCM_List *pair = new SCM_List();
        pair->data = sym;
        pair->next = nullptr;
        it->next = pair;
        it = pair;
        ret = cdr(ret);
      }
      else {
        printf("%s:%d [%s] not supported %s\n", __BASE_FILE__, __LINE__, __func__, car(ret).to_std_string().c_str());
        std::exit(1);
      }
    }
  }
  if (dummy.next) {
    SCM *l = new SCM();
    l->type = SCM::LIST;
    l->value = dummy.next;
    if (debug_enabled) {
      printf("-->");
      print_ast(l);
      printf("\n");
    }
    return l;
  }
  else {
    return scm_nil();
  }
}

SCM *parse(const char *s) {
  UString unicode(s);
  Parser parser(unicode);
  auto ret = parser.parse();
  return translate(ret);
}

SCM_List *parse_file(const char *filename) {
  auto res = read_file(filename);
  if (!std::holds_alternative<UString>(res)) {
    return nullptr;
  }
  SCM_List dummy_list;
  dummy_list.data = nullptr;
  dummy_list.next = nullptr;
  auto it = &dummy_list;
  auto code = std::get<UString>(res);
  Parser parser(code, filename);
  Cell ast = parser.next();
  while (!ast.is_none()) {
    Cell ret;
    auto expr = translate(ast);
    it->next = make_list(expr);
    it = it->next;
    ast = parser.next();
  }
  return dummy_list.next;
}