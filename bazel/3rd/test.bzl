load("@rules_cc//cc:defs.bzl", "cc_test")

PSCM_TEST_COPTS = [
    "-std=c++20",
]

PSCM_TEST_DEPS = [
    ":pscm",
    "@doctest//doctest",
]
MODE = struct(
    DIRECT = "DIRECT",
    REGISTER_MACHINE = "REGISTER_MACHINE",
)

def _pscm_main_test_impl(ctx):
    cwd = ctx.attr.cwd
    args = ctx.attr.args
    output_file = ctx.outputs.output_file
    substitutions = {
        "%{cwd}": cwd,
        "%{args}": " ".join(args),
    }
    ctx.actions.expand_template(
        template = ctx.file._template,
        substitutions = substitutions,
        output = output_file,
    )
    return DefaultInfo(
        runfiles = ctx.runfiles(files = ctx.files.data),
        executable = output_file,
    )

pscm_main_test = rule(
    implementation = _pscm_main_test_impl,
    attrs = {
        "cwd": attr.string(),
        "_template": attr.label(
            default = "test/pscm_main.sh.tpl",
            allow_single_file = True,
        ),
        "data": attr.label_list(
            allow_files = [""],
        ),
        "output_file": attr.output(mandatory = True),
    },
    output_to_genfiles = True,
    test = True,
)

def _gen_pscm_main_test(cwd, mode, entry, other = []):
    if mode != "DIRECT" and mode != "REGISTER_MACHINE":
        fail("not supported mode: ", mode)
    name = entry + "_" + mode + "_test"
    path_arr = cwd.split("/")
    path = "_".join(path_arr)
    if len(path) > 0:
        path += "-"
    pscm_main_test(
        name = path + name,
        cwd = cwd,
        args = ["-m", mode, "-s", entry],
        data = [cwd + "/" + entry, ":pscm-main"] + other,
        output_file = path + name + ".sh",
        size = "small",
    )

def _collect_sicp_tests():
    cc_test(
        name = "sicp_ch1_tests",
        srcs = ["test/sicp/ch1_tests.cpp"],
        copts = PSCM_TEST_COPTS,
        deps = PSCM_TEST_DEPS,
        size = "small",
    )

def _collect_r4rs_tests():
    cc_test(
        name = "r4rs_test",
        srcs = ["test/r4rs/r4rs_tests.cpp"],
        copts = PSCM_TEST_COPTS,
        deps = PSCM_TEST_DEPS,
        size = "small",
    )
    _gen_pscm_main_test("test/r4rs", MODE.DIRECT, "r4rstest.scm")
    _gen_pscm_main_test("test/r4rs", MODE.REGISTER_MACHINE, "r4rstest.scm")
    _gen_pscm_main_test("test/r4rs", MODE.REGISTER_MACHINE, "r4rs_cont_test.scm")
    _gen_pscm_main_test(
        "test/r4rs",
        MODE.DIRECT,
        "load.scm",
        ["test/r4rs/init.scm", "test/r4rs/r4rstest.scm"],
    )
    _gen_pscm_main_test(
        "test/r4rs",
        MODE.REGISTER_MACHINE,
        "load.scm",
        ["test/r4rs/init.scm", "test/r4rs/r4rstest.scm"],
    )

def _collect_r5rs_tests():
    _gen_pscm_main_test("test/r5rs", MODE.DIRECT, "r5rstest.scm")
    _gen_pscm_main_test("test/r5rs", MODE.REGISTER_MACHINE, "r5rstest.scm")
    _gen_pscm_main_test(
        "test/r5rs",
        MODE.DIRECT,
        "load.scm",
        ["test/r4rs/init.scm", "test/r5rs/r5rstest.scm"],
    )
    _gen_pscm_main_test(
        "test/r5rs",
        MODE.REGISTER_MACHINE,
        "load.scm",
        ["test/r4rs/init.scm", "test/r5rs/r5rstest.scm"],
    )

def _collect_module_tests():
    cc_test(
        name = "module_test",
        srcs = ["test/module/load_path_tests.cpp"],
        copts = PSCM_TEST_COPTS,
        deps = PSCM_TEST_DEPS,
        size = "small",
    )
    _gen_pscm_main_test(
        "test/module",
        MODE.DIRECT,
        "r5rs_test.scm",
        ["test/module/test.scm"],
    )
    _gen_pscm_main_test(
        "test/module/texmacs",
        MODE.DIRECT,
        "init.scm",
        [
            "test/module/texmacs/boot.scm",
            "test/module/texmacs/m.scm",
        ],
    )

def collect_pscm_tests():
    test_list = native.glob(["test/*_tests.cpp"])
    for item in test_list:
        name = item.split("/")[-1]
        if name.endswith("s.cpp"):
            name = name[:-5]
        cc_test(
            name = name,
            srcs = [item],
            copts = PSCM_TEST_COPTS,
            deps = PSCM_TEST_DEPS,
            size = "small",
        )
    _collect_sicp_tests()
    _collect_r4rs_tests()
    _collect_r5rs_tests()
    _collect_module_tests()
