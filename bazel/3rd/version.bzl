"""
Executes a python script that gets the volatile & stable workspace status
files as an input.

The result is a C++ header & source file containing the git hash & git dirty flag.
"""

def _impl(ctx):
    # The list of arguments we pass to the script.
    # volatile status file: ctx.version_file
    # stable status file: ctx.info_file
    args = []
    args += ["--source", ctx.outputs.source_file.path]
    args += ["--volatile_file", ctx.version_file.path]
    args += ["--stable_file", ctx.info_file.path]
    args += ["--commit_hash_name", ctx.attr.commit_variable_name]
    args += ["--git_branch_name", ctx.attr.branch_variable_name]
    args += ["--workspace_dirty_name", ctx.attr.dirty_variable_name]
    args += ["--version_major", ctx.attr.version_major]
    args += ["--version_minor", ctx.attr.version_minor]
    args += ["--version_patch", ctx.attr.version_patch]

    # Action to call the script.
    ctx.actions.run(
        inputs = [ctx.version_file, ctx.info_file],
        outputs = [ctx.outputs.source_file],
        arguments = args,
        progress_message = "Adding version info to %s" % ctx.outputs.source_file.short_path,
        executable = ctx.executable._gen_cpp_tool,
    )

def gen_pscm_version_info(**kwargs):
    _gen_pscm_version_info(
        **kwargs
    )

_gen_pscm_version_info = rule(
    implementation = _impl,
    attrs = {
        "version_major": attr.string(mandatory = True),
        "version_minor": attr.string(mandatory = True),
        "version_patch": attr.string(mandatory = True),
        "_template": attr.label(
            default = "include/pscm/version.h.in",
            allow_single_file = True,
        ),
        "source_file": attr.output(mandatory = True),
        "_gen_cpp_tool": attr.label(
            executable = True,
            cfg = "host",
            allow_files = True,
            default = Label("//:gen_cpp"),
        ),
        "commit_variable_name": attr.string(mandatory = True),
        "branch_variable_name": attr.string(mandatory = True),
        "dirty_variable_name": attr.string(mandatory = True),
    },
    output_to_genfiles = True,
)
