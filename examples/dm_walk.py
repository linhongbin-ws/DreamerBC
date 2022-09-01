import dreamerv2.api as dv2
import argparse
from dreamerv2 import common


config = dv2.defaults.update({
  'logdir': './data/dmc_walker_walk_async',
  'task': 'dmc_walker_walk',
  'encoder': {'mlp_keys': '$^', 'cnn_keys': 'image'},
  'decoder': {'mlp_keys': '$^', 'cnn_keys': 'image'},
  'action_repeat': 2,
  'eval_every': 1e4,
  'prefill': 1000,
  'pretrain': 100,
  'clip_rewards': 'identity',
  'pred_discount': False,
  'replay.prioritize_ends': False,
  'grad_heads': ['decoder', 'reward'],
  'rssm': {'hidden': 200, 'deter': 200},
  'model_opt.lr': 3e-4,
  'actor_opt.lr': 8e-5,
  'critic_opt.lr': 8e-5,
  'actor_ent': 1e-4,
  'kl.free': 1.0,
}).parse_flags()


suite, task = config.task.split('_', 1)
env = common.DMC(
    task, config.action_repeat, config.render_size, config.dmc_camera)
env = common.GymWrapper(env)
env = common.ResizeImage(env)
env = common.NormalizeAction(env)
env = common.TimeLimit(env, config.time_limit)
dv2.train(env, config, is_train=config.istrain, skip_gym_wrap=True)




env.close()
