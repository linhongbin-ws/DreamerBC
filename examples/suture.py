from gym_suture.envs.wrapper import make_env
import dreamerv2.api as dv2
import argparse


config = dv2.defaults.update({
  'bc_dir': './data/ambf_np_green_sparse_async/train_episodes/oracle',
  'logdir': './data/ambf_np_green_sparse_async',
  'log_every': 1e3,
  'train_every': 2,
  # 'loss_scales.kl': 1.0,
  
  'discount': 0.999,
  'task': 'suture_ambf_np',
  'encoder': {'mlp_keys': '$^', 'cnn_keys': 'image'},
  'decoder': {'mlp_keys': '$^', 'cnn_keys': 'image'},
  # 'action_repeat': 1,
  'eval_every': 2e3,
  'pretrain': 100,
  'clip_rewards': 'identity',
  # 'pred_discount': True,
  # 'grad_heads': ['decoder', 'reward','discount'],
  # 'rssm': {'hidden': 200, 'deter': 200, 'stoch': 32, 'discrete': 32},
  'model_opt.lr': 2e-4,
  'actor_opt.lr': 4e-5,
  # 'actor_opt.clip': 1e3,
  'critic_opt.lr': 1e-4,
  'actor_ent': 2e-3,
  # 'kl.free': 1.0,
  'prefill': 0,
  # 'prefill': 30000,
  'prefill_agent': 'oracle',
  'time_limit': 50,
  'replay': {'capacity': 2e6, 'ongoing': False, 'minlen': 17, 'maxlen': 17, 'prioritize_ends': False},
  'dataset': {'batch': 16, 'length': 17},
  'reward_norm_skip': True,
  # 'expl_noise': 0.1
  
  #### debug
  # 'jit': False,
  # 'replay.capacity': 2e4,
}).parse_flags()




env = make_env("ambf_needle_picking_64x64_discrete")
dv2.train(env, config, is_train=config.istrain)

env.close()
