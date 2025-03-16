import lit.util
from lit.llvm.subst import FindTool, ToolSubst

class PSCMConfig(object):
    def __init__(self, lit_config, config):
        self.lit_config = lit_config
        self.config = config