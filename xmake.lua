add_requires("spdlog", {configs = {header_only = false}})
add_requires("doctest")
add_requires("universal_stacktrace")
add_requires("cpp-linenoise")
set_version("0.3.0")
option("cxx20-modules")
    set_description("enable c++20 modules")
    add_defines("PSCM_USE_CXX20_MODULES")
    set_default(false)
    set_values(false, true)
option_end()
target("pscm") do
    set_kind("static")
    add_options("cxx20-modules")
    add_configfiles(
        "src/version.cpp.xmake", {
            filename = "version.cpp"})
    set_languages("cxx20", "c17")
    add_includedirs("include")
    add_packages({"spdlog","universal_stacktrace","cpp-linenoise"})
    add_files({
        "src/**.cpp",
        "$(buildir)/version.cpp"})
    if has_config("cxx20-modules") then
        add_files({
            "src/**.cppm",
            "3rd/std.cppm",
            "3rd/fmt.cppm",})
    end
end

for _, filepath in ipairs(os.files("test/**_tests.cpp")) do
    local testname = path.basename(filepath) 
    target(testname) do 
        add_options("cxx20-modules")
        set_group("tests")
        add_deps("pscm")
        add_packages({"doctest","spdlog","universal_stacktrace"})
        set_languages("cxx20", "c17")

        add_includedirs("include")
        add_files(filepath)
    end
end