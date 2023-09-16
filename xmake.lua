add_rules("mode.debug", "mode.releasedbg")
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
add_requires("icu4c")
add_requires("mscharconv")
set_version("0.3.0")
target("pscm") do
    set_kind("static")
    add_options("cxx20-modules")
    add_configfiles(
        "src/version.cpp.xmake", {
            filename = "version.cpp"})
    set_languages("cxx20")
    add_includedirs("include", {public = true})
    add_headerfiles("include/**.h")
    add_packages({"spdlog", "universal_stacktrace", "cpp-linenoise"})
    add_packages({"icu4c", "mscharconv"}, {public = true})
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
target("repl") do
    set_kind("binary")
    add_options("cxx20-modules")
    set_languages("cxx20")
    add_deps("pscm")
    add_files("main.cpp")
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

        add_files(filepath)
        
        if is_mode ("coverage") then
            add_cxxflags("-O0")
            add_cxxflags("-fprofile-arcs")
            add_cxxflags("-ftest-coverage")
            add_ldflags("-coverage")
        end
    end
end

local integrated_tests = {
    {"test/r4rs/r4rstest.scm", "DIRENT"},
    {"test/r4rs/r4rstest.scm", "REGISTER_MACHINE"},
    {"test/r4rs/r4rs_cont_test.scm", "REGISTER_MACHINE"},
    {"test/r4rs/load.scm", "DIRENT"},
    {"test/r5rs/r5rstest.scm", "DIRENT"},
    {"test/r5rs/r5rstest.scm", "REGISTER_MACHINE"},
    {"test/r5rs/load.scm", "DIRENT"},
    {"test/r5rs/load.scm", "REGISTER_MACHINE"},
    {"test/module/r5rs_test.scm", "DIRENT"},
    {"test/module/texmacs/init.scm", "DIRENT"},
}

for _, entry in ipairs(integrated_tests) do
    local filepath = entry[1]
    local mode = entry[2]
    local testname = table.concat(
        table.join2(table.slice(path.split(filepath), 3), {mode}),
        "_")
    target(testname) do 
        add_options("cxx20-modules")
        set_kind("phony")
        set_group("tests")
        add_deps("repl")
        on_run(function (target)
            import("core.base.option")
            import("core.project.project")
            os.cd(path.directory(filepath))
            local dep = project.target("repl")
            local exec = dep:targetfile()
            print(testname .. " start!")
            
            local args = {
                "-m", mode,
                "-s", path.filename(filepath)
            }
            -- debugging?
            if option.get("debug") then
                import("devel.debugger")
                debugger.run(exec, args)
            else
                os.iorunv(exec, args)
                print(testname .. " success!")
            end
        end)
        add_packages({"doctest","spdlog","universal_stacktrace"})
    end
end
