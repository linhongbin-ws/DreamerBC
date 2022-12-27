import dreamerv2.api as dv2

config = dv2.defaults.update({})

config.save('default_config.yaml')