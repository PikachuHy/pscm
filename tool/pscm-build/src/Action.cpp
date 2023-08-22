module;
#include <pscm/common_def.h>
module pscm.build;
import :Action;
import std;
import fmt;
import subprocess;
import pscm.logger;
import pscm.compat;

namespace pscm::build {

void make_sure_parent_path_exist(const std::string& filename);

void Action::run(std::string_view cwd) {
  PSCM_INLINE_LOG_DECLARE("pscm.build.Action");
  // create directories first
  // fix No such file or directory
  auto output_file = get_output_file();
  if (output_file.has_value()) {
    make_sure_parent_path_exist(fmt::format("{}/{}", cwd, output_file.value()));
  }
  auto command_line = get_args();
  PSCM_DEBUG("RUN: {}", command_line);
  auto p = subprocess::run(command_line, subprocess::RunBuilder().cwd(std::string(cwd)));
  if (p.returncode != 0) {
    PSCM_ERROR("ERROR: return code: {}", p.returncode);
    std::exit(1);
  }
}

} // namespace pscm::build