#include <pscm_c_api.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace {

struct Scheme {
  Scheme() {
    scm = pscm_create_scheme();
  }

  ~Scheme() {
    pscm_destroy_scheme(scm);
  }

  const char *eval(const char *code) {
    auto ret = pscm_eval(scm, code);
    return pscm_to_string(ret);
  }

private:
  void *scm;
};

} // namespace

PYBIND11_MODULE(pypscm, m) {
  py::class_<Scheme>(m, "Scheme").def(py::init<>()).def("eval", &Scheme::eval);
}