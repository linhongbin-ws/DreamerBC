from gym_suture.envs.wrapper import make_env
import dreamerv2.api as dv2
import argparse


config = dv2.defaults.update({
  # 'bc_dir': './data/ambf_np_seg_depth_sparse/train_episodes/oracle',
  # 'logdir': './data/ambf_np_seg_depth_sparse',

  'bc_dir': './data/ambf_np_seg_depth_gripperstate_sparse/train_episodes/oracle',
  'logdir': './data/ambf_np_seg_depth_gripperstate_sparse',
  
  'log_every': 4e2,
  # 'loss_scales.kl': 1.0,
  
  'discount': 0.999,
  'task': 'suture_ambf_np',
  'encoder': {'mlp_keys': '$^', 'cnn_keys': 'image'},
  'decoder': {'mlp_keys': '$^', 'cnn_keys': 'image'},
  # 'action_repeat': 1,
  'eval_every': 2e3,
  'pretrain': 100,
  'clip_rewards': 'identity',
  # 'grad_heads': ['decoder', 'reward','discount'],
  # 'rssm': {'hidden': 200, 'deter': 200, 'stoch': 32, 'discrete': 32},
  'model_opt.lr': 1e-4,
  'actor_opt.lr': 4e-6,
  'critic_opt.lr': 1e-5,
  'actor_ent': 2e-3,
  'prefill': 1e4,
  'prefill_agent': 'oracle',
  'time_limit': 50,
  'replay': {'capacity': 2e6, 'ongoing': False, 'minlen': 17, 'maxlen': 17, 'prioritize_ends': False},
  'dataset': {'batch': 16, 'length': 17},
  'reward_norm_skip': True,
  # 'grad_extra_image_channel_scale': [0,3,0], # rgb, emphasis green
  'train_every': 50,
  'train_steps': 40,
  
  # #### debug
  # 'jit': False,
  # 'replay.capacity': 2e4,
  # 'log_every': 10,
  # 'prefill': 100,
}).parse_flags()



env = make_env('ambf_needle_picking_64x64_discrete',is_segment_filter=True, is_gripper_state_image=True)
dv2.train(env, config, is_pure_train=config.is_pure_train, is_pure_datagen=config.is_pure_datagen)

env.close()
