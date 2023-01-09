from gym_suture.envs.wrapper import make_env
import dreamerv2.api as dv2
from pathlib import Path
import ruamel.yaml as yaml
from dreamerv2 import common
import pathlib
import argparse

#========================================
parser = argparse.ArgumentParser()
# RL related
parser.add_argument('--json', type=str, default="")
parser.add_argument('--section', type=int, required=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--eval-eps', type=int, default=20)
parser.add_argument('--baseline', type=str, default="DreamerBC")
parser.add_argument('--only-train', action='store_true')
parser.add_argument('--only-datagen', action='store_true')
parser.add_argument('--prefill', type=int, default=-1) # <0 means following default settings

# env related
parser.add_argument('--robot', type=str, default='ambf') # [ambf, dvrk]
parser.add_argument('--arm', type=str, default='psm2') # [psm1, psm2]
parser.add_argument('--preprocess-type', type=str, default='segment_script') # [segment_net, mixdepth,origin, segment_script]
parser.add_argument('--preprocess-method', type=str, default='zoom_needle_gripper_boximage') #[zoom_needle_gripper_boximage, zoom_needle_boximage]
parser.add_argument('--clutch', type=int, default=6)

args = parser.parse_args()

#==================================

section = args.section
baseline = args.baseline
logdir = str(Path('./data/suture/needle_picking/ambf') / baseline / args.preprocess_type / args.preprocess_method / str(section) )
# in_json_path = pathlib.Path(args.json)

configs = yaml.safe_load((pathlib.Path("./examples/jsons/default_np.yaml")).read_text())
config = common.Config(configs)
if args.json is not "":
  configs = yaml.safe_load((pathlib.Path(args.json)).read_text())
  config = config.update(configs)

config = config.update({
  'bc_dir': logdir + '/train_episodes/oracle',
  'logdir': logdir,         
  'eval_eps': args.eval_eps,
  'seed': args.seed,
                 })

assert args.baseline in ["DreamerBC", "Dreamer"]
if args.baseline == "Dreamer":
  config=config.update({
    "bc_grad_weight": 0,
    "bc_wm_retrain": False,
    "bc_agent_retrain": False,
  })

if args.prefill>=0:
  config=config.update({
    "prefill": args.prefill
  })
print(config)

# if config.is_pure_train:
#   env = None
# else:
# env = make_env('ambf_needle_picking_64x64_discrete',  
#                     is_visualizer=True, 
#             image_preprocess_type=image_preprocess_type,
#                image_preprocess_method=args.seg_method,
#             is_depth=True, 
#             is_idle_action=False,
#             is_ds4_oracle=False,
#             action_arm_device='psm2',
#             obs_type="image",
#             timelimit=None,
#             resize_resolution=64,
#             is_dummy=config.is_pure_train,
            
env = make_env(task="{}_needle_picking_64x64_discrete".format(args.robot),
             image_preprocess_type=args.preprocess_type if args.preprocess_type not in ["origin"] else None, 
             image_preprocess_method=args.preprocess_method,
            #  scalar2image_obs_key=["gripper_state", "state"],
             action_arm_device=args.arm,
             clutch_start_engaged=args.clutch,
             resize_resolution=64,
            #  timelimit=-1, 
            #  is_depth=True, 
            #  is_idle_action=False,
             is_ds4_oracle=False,
             is_visualizer=False,
            #  is_visualizer_blocking=True, 
             is_dummy=config.is_pure_train,
)
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
