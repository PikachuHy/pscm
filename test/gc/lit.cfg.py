import lit.formats

config.name = 'pscm GC Tests'

config.test_format = lit.formats.ShTest(True)

config.test_source_root = os.path.dirname(__file__)

config.suffixes = ['.scm']

config.environment["PATH"] = "/usr/local/opt/llvm/bin:" + config.environment["PATH"]
