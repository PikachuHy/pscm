//
// Created by PikachuHy on 2023/7/21.
//
module;
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <ostream>
#include <queue>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#ifdef __GNUC__
# 3 __FILE__ 1 3 // Enter "faked" system files since std is reserved module name
#endif
export module std;

export namespace std {
using std::cout;
using std::endl;
using std::fstream;
using std::map;
using std::ostream;
using std::queue;
using std::size_t;
using std::string;
using std::string_view;
using std::tuple;
using std::unordered_map;
using std::unordered_set;
using std::vector;
using std::operator+;
using std::operator<;
using std::operator>;
using std::operator==;
using std::operator<<;
using std::operator>>;
using std::gcd;
using std::hash;
using std::lcm;
using std::move;
using std::optional;
using std::span;
using std::thread;

using std::filesystem::path;

inline namespace string_literals {
using std::literals::operator""s;
}
} // namespace std