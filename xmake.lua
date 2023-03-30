add_requires("spdlog")
add_requires("doctest")
target("pscm") do
    set_kind("static")
    set_languages("c++20")
    add_includedirs("include")
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
        add_files(filepath)
    end
end