def gen_pscm_version_header(**kwargs):
    _gen_pscm_version_header(
        **kwargs
    )

def _gen_pscm_version_header_impl(ctx):
    ctx.actions.expand_template(
        template = ctx.file._template,
        output = ctx.outputs.header_file,
        substitutions = {
            "@GIT_BRANCH@": ctx.attr.git_branch,
            "@GIT_HASH@": ctx.attr.git_hash,
            "@pscm_VERSION@": ctx.attr.version_major + "." + ctx.attr.version_minor + "." + ctx.attr.version_patch,
            "@pscm_VERSION_MAJOR@": ctx.attr.version_major,
            "@pscm_VERSION_MINOR@": ctx.attr.version_minor,
            "@pscm_VERSION_PATCH@": ctx.attr.version_patch,
        },
    )

_gen_pscm_version_header = rule(
    implementation = _gen_pscm_version_header_impl,
    attrs = {
        "git_branch": attr.string(mandatory = True),
        "git_hash": attr.string(mandatory = True),
        "version_major": attr.string(mandatory = True),
        "version_minor": attr.string(mandatory = True),
        "version_patch": attr.string(mandatory = True),
        "_template": attr.label(
            default = "include/pscm/version.h.in",
            allow_single_file = True,
        ),
        "header_file": attr.output(mandatory = True),
    },
    output_to_genfiles = True,
)
