from gym_suture.envs.wrapper import make_env
import dreamerv2.api as dv2
import argparse


config = dv2.defaults.update({
  'logdir': './data/ambf_np_green_async',
  'log_every': 1e3,
  'train_every': 2,
  'loss_scales.kl': 1.0,
  'discount': 0.99,
  'task': 'suture_ambf_np',
  'encoder': {'mlp_keys': '$^', 'cnn_keys': 'image'},
  'decoder': {'mlp_keys': '$^', 'cnn_keys': 'image'},
  'action_repeat': 1,
  'eval_every': 2e3,
  'prefill': 100,
  'pretrain': 100,
  'clip_rewards': 'identity',
  'pred_discount': True,
  'grad_heads': ['decoder', 'reward','discount'],
  'rssm': {'hidden': 200, 'deter': 200, 'stoch': 32, 'discrete': 32},
  'model_opt.lr': 1e-4,
  'actor_opt.lr': 8e-7,
  'actor_opt.clip': 1e2,
  'critic_opt.lr': 8e-7,
  'actor_ent': 1e-4,
  'kl.free': 1.0,
  'prefill': 1000,
  'prefill_agent': 'oracle',
  'time_limit': 50,
  'replay': {'capacity': 2e6, 'ongoing': False, 'minlen': 16, 'maxlen': 50, 'prioritize_ends': True},
  'dataset': {'batch': 16, 'length': 17},
  'jit': False,
  'actor_type': "AlphaZero",
  'train_only_wm_steps': 0,
  'train_mcts_c_puct': 3,
  'train_mcts_n_playout': 80,
}).parse_flags()




env = make_env("ambf_needle_picking_64x64_discrete")
dv2.train(env, config, is_train=config.istrain)

env.close()
