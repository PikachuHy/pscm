//
// Created by PikachuHy on 2023/7/21.
//
module;
#include <linenoise.hpp>
export module linenoise;

export namespace linenoise {
using linenoise::AddHistory;
using linenoise::LoadHistory;
using linenoise::Readline;
using linenoise::SaveHistory;
using linenoise::SetCompletionCallback;
using linenoise::SetHistoryMaxLen;
using linenoise::SetMultiLine;
} // namespace linenoise
