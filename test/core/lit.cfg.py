import lit.formats

config.name = 'pscm JIT Tests'

config.test_format = lit.formats.ShTest(True)

config.test_source_root = os.path.dirname(__file__)

config.suffixes = ['.ll', '.scm']

config.environment["PATH"] = "/usr/local/opt/llvm/bin:" + config.environment["PATH"]
