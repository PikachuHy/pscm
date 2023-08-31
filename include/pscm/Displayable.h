//
// Created by PikachuHy on 2023/3/12.
//

#pragma once
#include "unicode/unistr.h"
#include "unicode/ustream.h"
#include "pscm/misc/ICUCompat.h"
#include <iostream>
#include <type_traits>
#include <utility>

namespace pscm {
  
  template <typename T>
  concept DisplayableObj = requires(T a) {
    {a.to_string()} -> std::same_as<UString>;
  };
  template <DisplayableObj T>
  const UString to_string(T obj){
    return obj.to_string();
  };

  template <typename T>
  concept Displayable = requires(T a) {
    {pscm::to_string(a)} -> std::same_as<const UString>;
  };
} // namespace pscm
