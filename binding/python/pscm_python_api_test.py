import pypscm

import sys


# FIXME: how to use pip?
# Currently, rules_python version is not correct
# import pytest


def test_scheme_add():
    scm = pypscm.Scheme()
    ret = scm.eval("(+ 2 6)")
    assert ret == "8"


# if using 'bazel test ...'
if __name__ == "__main__":
    test_scheme_add()
