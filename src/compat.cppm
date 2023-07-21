//
// Created by PikachuHy on 2023/7/22.
//
module;
#include "pscm/compat.h"
#if PSCM_STD_COMPAT
#include <ghc/filesystem.hpp>
#else
#include <filesystem>
#endif
export module pscm.compat;

export namespace pscm {
using pscm::StringView;
}
#if PSCM_STD_COMPAT
export namespace fs = ghc::filesystem;
#else
export namespace fs = std::filesystem;
#endif