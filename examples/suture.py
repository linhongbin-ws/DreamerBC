from gym_suture.envs.wrapper import make_env
import dreamerv2.api as dv2
import argparse


config = dv2.defaults.update({
  # 'bc_dir': './data/ambf_np_seg_depth_sparse/train_episodes/oracle',
  # 'logdir': './data/ambf_np_seg_depth_sparse',

  'bc_dir': './data/ambf_np_seg_depth_gripperstate2/train_episodes/oracle',
  'logdir': './data/ambf_np_seg_depth_gripperstate2',
  
  'log_every': 4e2,
  # 'loss_scales.kl': 1.0,
  
  'discount': 0.999,
  # 'discount_lambda': 0.0,
  # 'slow_target': False,
  'task': 'suture_ambf_np',
  'encoder': {'mlp_keys': '$^', 'cnn_keys': 'image'},
  'decoder': {'mlp_keys': '$^', 'cnn_keys': 'image'},
  # 'action_repeat': 1,
  'rssm': {'hidden': 200, 'deter': 200},
  'pretrain': 100,
  'clip_rewards': 'identity',
  # 'grad_heads': ['decoder', 'reward','discount'],
  # 'rssm': {'hidden': 200, 'deter': 200, 'stoch': 32, 'discrete': 32},
  'model_opt.lr': 1e-4,
  # 'actor_opt.lr': 4e-5,
  # 'actor_opt.lr': 0.0,
  'critic_opt.lr': 4e-5,
  'actor_opt.lr': 2e-5,
  # 'critic_opt.lr': 0,
  # 'critic.layers': 5,
  # 'actor_ent': 2e-3,
  # 'actor_ent': 2e-7,
  'actor_ent': 1e-7,
  'prefill': 8e3,
  # 'prefill': 1e1,
  'prefill_agent': 'oracle',
  'time_limit': 50,
  'replay': {'capacity': 2e6, 'ongoing': False, 'minlen': 10, 'maxlen': 10, 'prioritize_ends': False},
  'dataset': {'batch': 70, 'length': 10},
  'reward_norm_skip': True,
  # 'grad_extra_image_channel_scale': [0,3,0], # rgb, emphasis green
  'train_every': 50,
  'train_steps': 50,
  'eval_eps': 58,
  'eval_every': 8e2,
  'bc_grad_weight': 10,
  'save_sucess_eps_rate': 0.7,
  'bc_skip_start_step_num': 2,
  # 'slow_target_update': 150
  # #### debug
  # 'jit': False,
  # 'replay.capacity': 2e4,
  # 'log_every': 200,
  # 'eval_eps': 1,
  # 'prefill': 100,
  # 'eval_every': 100,
  # 'train_steps': 60,
  # 'train_every': 1000000,
  # 'save_sucess_eps_rate': 0.0,
}).parse_flags()



env = make_env('ambf_needle_picking_64x64_discrete',is_segment_filter=True, is_gripper_state_image=True,is_idle_action=False)
dv2.train(env, config, is_pure_train=config.is_pure_train, is_pure_datagen=config.is_pure_datagen)

env.close()
