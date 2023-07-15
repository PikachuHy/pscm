import argparse, sys, os


class VariableStore:
    def __init__(self, values, is_reliable, name_prefix):
        self.values = values
        self.name_prefix = name_prefix
        self.is_reliable = is_reliable

    def get(self, name):
        return self.values.get(self.name_prefix + name)


class MultipleVariableStore:
    def __init__(self):
        self.values = []

    def add_file(self, path, is_reliable, name_prefix=""):
        result = {}
        with open(path, "r") as f:
            for entry in f.read().split("\n"):
                if entry:
                    key_value = entry.split(' ', 1)
                    key = key_value[0].strip()
                    if key in result:
                        sys.stderr.write("Error: Duplicate key '{}'\n".format(key))
                        sys.exit(1)
                    else:
                        result[key] = key_value[1].strip()
        self.values.append(VariableStore(result, is_reliable, name_prefix))

    def get(self, name):
        for l in self.values:
            result = l.get(name)
            if result is not None:
                return (result, l.is_reliable)
        return (None, False)


def setup_path(file_path):
    header_dir = os.path.normpath(os.path.join(file_path, ".."))
    if not os.path.exists(header_dir):
        os.makedirs(header_dir)


def main():
    parser = argparse.ArgumentParser(description='write version info to source file')
    parser.add_argument('--source',
                        required=True,
                        help='output source file')
    parser.add_argument('--volatile_file',
                        required=True,
                        help='file containing the volatile variables')
    parser.add_argument('--stable_file',
                        required=True,
                        help='file containing the stable variables')
    parser.add_argument('--commit_hash_name',
                        help='variablename of the hash')
    parser.add_argument('--git_branch_name',
                        help='variablename of git branch')
    parser.add_argument('--version_major')
    parser.add_argument('--version_minor')
    parser.add_argument('--version_patch')
    parser.add_argument('--workspace_dirty_name',
                        help='variablename of the boolean communicating if the workspace has no local changes')

    args = parser.parse_args()

    variables = MultipleVariableStore()
    variables.add_file(args.stable_file, True, "STABLE_")
    variables.add_file(args.volatile_file, False)

    (commit_hash, commit_hash_reliable) = variables.get(args.commit_hash_name.strip())
    (git_branch, git_branch_reliable) = variables.get(args.git_branch_name.strip())
    (is_dirty_str, is_dirty_reliable) = variables.get(args.workspace_dirty_name.strip())
    is_dirty = "0" != is_dirty_str
    pscm_version = "{version_major}.{version_minor}.{version_patch}".format(
        version_major=args.version_major,
        version_minor=args.version_minor,
        version_patch=args.version_patch,
    )
    setup_path(args.source)
    with open(args.source, "w") as f:
        f.write("""
#include "pscm/version.h"
namespace pscm {{
const char* VersionInfo::GIT_HASH = "{git_hash}";
const char* VersionInfo::GIT_BRANCH = "{git_branch}";
const char* VersionInfo::PSCM_VERSION = "{pscm_version}";
const int VersionInfo::PSCM_VERSION_MAJOR = {version_major};
const int VersionInfo::PSCM_VERSION_MINOR = {version_minor};
const int VersionInfo::PSCM_VERSION_PATCH = {version_patch};
}}
""".format(
            git_hash=commit_hash,
            git_branch=git_branch,
            pscm_version=pscm_version,
            version_major=args.version_major,
            version_minor=args.version_minor,
            version_patch=args.version_patch,
        )
        )


if __name__ == "__main__":
    main()
