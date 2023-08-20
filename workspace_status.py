import subprocess, sys, os


def main():
    git_hash = get_git_hash(".")
    git_branch = get_git_branch(".")
    git_is_dirty = is_git_dirty(".")

    print("STABLE_GIT_COMMIT_HASH {}".format(git_hash))
    print("STABLE_GIT_BRANCH {}".format(git_branch))
    print("STABLE_GIT_DIRTY {}".format("1" if git_is_dirty else "0"))
    print("STABLE_DUMMY A")


def get_git_hash(path):
    cmd = ["git", "rev-parse", "HEAD"]
    p = subprocess.Popen(cmd, cwd=path, stdout=subprocess.PIPE)
    (out, err) = p.communicate()
    if p.returncode != 0:
        print(f"run {cmd} fail")
        sys.exit(p.returncode)
    return out.decode("ascii").strip()


# symbolic-ref --short -q HEAD
def get_git_branch(path):
    cmd = ["git", "symbolic-ref", "--short", "-q", "HEAD"]
    p = subprocess.Popen(cmd, cwd=path, stdout=subprocess.PIPE)
    (out, err) = p.communicate()
    if p.returncode != 0:
        print(f"run {cmd} fail")
        # sys.exit(p.returncode)
        # can not get branch when run on GitHub PR ci
        return ""
    return out.decode("ascii").strip()


def is_git_dirty(path):
    cmd = ["git", "status", "-s"]
    p = subprocess.Popen(cmd, cwd=path, stdout=subprocess.PIPE)
    (out, err) = p.communicate()
    if p.returncode != 0:
        print(f"run {cmd} fail")
        sys.exit(p.returncode)
    return not not out.decode("ascii").strip()


if __name__ == "__main__":
    main()
