from pscm.lit import config
pscm_config = None

def initialize(lit_config, test_config):
    global pscm_config
    pscm_config = config.PSCMConfig(lit_config, test_config)