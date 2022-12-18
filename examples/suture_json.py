from gym_suture.envs.wrapper import make_env
import dreamerv2.api as dv2
from pathlib import Path
import ruamel.yaml as yaml
from dreamerv2 import common
import pathlib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json', type=str, required=True)
parser.add_argument('--section', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--eval-eps', type=int, default=58)
parser.add_argument('--seg-proc', type=str, default="segment_script")
parser.add_argument('--baseline', type=str, default="DreamerBC")
parser.add_argument('--only-train', action='store_true')
parser.add_argument('--only-datagen', action='store_true')
args = parser.parse_args()

section = args.section
baseline = args.baseline
image_preprocess_type = args.seg_proc
logdir = str(Path('./data/suture/needle_picking/ambf') / baseline / image_preprocess_type / str(section) )
in_json_path = pathlib.Path(args.json)

#=========================================
# in_json_path = pathlib.Path(__file__).parent / 'jsons'/ 'dreamer2suture_1.yaml'

configs = yaml.safe_load((in_json_path).read_text())
defaults = common.Config(configs)
config = defaults.update({
  'bc_dir': logdir + '/train_episodes/oracle',
  'logdir': logdir,         
  ## debug        
    # 'jit': False,
  # 'replay.capacity': 2e4,
  # 'log_every': 200,
  'eval_eps': args.eval_eps,
  'seed': args.seed,
  # 'prefill': 100,
  # 'eval_every': 100,
  # 'train_steps': 60,
  # 'train_every': 1000000,
                 })


print(config)

# if config.is_pure_train:
#   env = None
# else:
env = make_env('ambf_needle_picking_64x64_discrete',  
                    is_visualizer=True, 
            image_preprocess_type=image_preprocess_type, 
            is_depth=True, 
            is_gripper_state_image=True,
            is_idle_action=False,
            is_ds4_oracle=False,
            action_arm_device='psm2',
            obs_type="image",
            timelimit=None,
            resize_resolution=64,
               is_dummy=config.is_pure_train,)
env.seed = args.seed
# # obs = env.reset()
# # import cv2
# # frame = cv2.resize(obs['image'], (1080, 1080), interpolation=cv2.INTER_AREA)
# # cv2.imshow('preview', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) # Display the resulting frame
# # k = cv2.waitKey(0)
# # for i in range(4):
# #   obs, reward, done, info = env.step(env.action_space.sample())
# #   frame = cv2.resize(obs['image'], (1080, 1080), interpolation=cv2.INTER_AREA)
# #   cv2.imshow('preview', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) # Display the resulting frame
# #   k = cv2.waitKey(0)
dv2.train(env, config, is_pure_train=args.only_train, is_pure_datagen=args.only_datagen)
# print(env.observation_space)
# print(env.action_space)
env.close()
