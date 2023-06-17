add_requires("spdlog")
add_requires("doctest")
set_version("0.2.0")
target("pscm") do
    set_kind("static")
    add_configfiles(
        "include/(pscm/version.h.xmake)", {
            filename = "version.h"})
    set_languages("c++20")
    add_includedirs("include")
    add_includedirs("3rd/UniversalStacktrace/ust")
    add_includedirs("3rd/cpp-linenoise")
    add_includedirs("$(buildir)")
    add_packages("spdlog")
    add_files("src/**.cpp")
end

for _, filepath in ipairs(os.files("test/**_tests.cpp")) do
    local testname = path.basename(filepath) 
    target(testname) do 
        set_group("tests")
        add_deps("pscm")
        add_packages({"doctest","spdlog"})
        set_languages("c++20")

        add_includedirs("include")
        add_includedirs("3rd/UniversalStacktrace/ust")
        add_files(filepath)
    end
end