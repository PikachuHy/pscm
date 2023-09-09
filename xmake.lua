option("cxx20-modules")
    set_description("enable c++20 modules")
    set_default(false)
    set_values(false, true)
option_end()

if has_config("cxx20-modules") then
    add_requires("spdlog", {configs = {
        header_only = false, fmt_external = true}})
    add_defines("PSCM_USE_CXX20_MODULES")
else
    add_requires("spdlog", {configs = {header_only = false}})
end
add_requires("doctest")
add_requires("universal_stacktrace")
add_requires("cpp-linenoise")
set_version("0.3.0")
target("pscm") do
    set_kind("static")
    add_options("cxx20-modules")
    add_configfiles(
        "src/version.cpp.xmake", {
            filename = "version.cpp"})
    set_languages("cxx20")
    add_includedirs("include")
    add_packages({"spdlog","universal_stacktrace","cpp-linenoise"})

    add_files({
        "src/**.cpp",
        "$(buildir)/version.cpp"})
    
    if is_mode("coverage") then
        add_cxxflags("-O0")
        add_cxxflags("-fprofile-arcs")
        add_cxxflags("-ftest-coverage")
        add_ldflags("-coverage")
    end
    
    if has_config("cxx20-modules") then
        add_files({
            "src/**.cppm",
            "3rd/std.cppm",
            "3rd/fmt.cppm",})
        -- dirty hack to include .cc file from fmt, which is source file and
        -- should not be included
        add_includedirs({
            "3rd/fmt/src",
        })
    end
end

---
--- coverage:
--- use `xmake f -m coverage` to enable coverage
--- first `rm -rf build/` to clean build cache
--- then `xmake build` to build
--- then `xmake run --group=tests` to run tests for coverage
--- run `lcov --directory . --capture --output-file cov/coverage.info`
--- run `genhtml cov/coverage.info --output-directory cov/coverage`
--- open `cov/coverage/index.html` in browser
--- 

for _, filepath in ipairs(os.files("test/**_tests.cpp")) do
    local testname = path.basename(filepath) 
    target(testname) do 
        add_options("cxx20-modules")
        set_group("tests")
        add_deps("pscm")
        add_packages({"doctest","spdlog","universal_stacktrace"})
        set_languages("cxx20")

        add_includedirs("include")
        add_files(filepath)
        
        if is_mode ("coverage") then
            add_cxxflags("-O0")
            add_cxxflags("-fprofile-arcs")
            add_cxxflags("-ftest-coverage")
            add_ldflags("-coverage")
        end
    end
end
