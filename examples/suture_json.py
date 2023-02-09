from gym_suture.envs.wrapper import make_env
import dreamerv2.api as dv2
import dreamerv2.api_eval as dv2_eval

from pathlib import Path
import ruamel.yaml as yaml
from dreamerv2 import common
import pathlib
import argparse

#========================================
parser = argparse.ArgumentParser()
# RL related
parser.add_argument('--json', type=str, default="")
parser.add_argument('--section', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--eval-eps', type=int, default=20)
parser.add_argument('--only-train', action='store_true')
parser.add_argument('--only-datagen', action='store_true')
parser.add_argument('--only-eval', action='store_true')
parser.add_argument('--prefill', type=int, default=8000) # <0 means following default settings
parser.add_argument('--timelimit', type=int, default=-1) # <0 means consistent with config file

# env related
parser.add_argument('--robot', type=str, default='ambf') # [ambf, dvrk]
parser.add_argument('--platform', type=str, default='phantom') #[cuboid, phantom]
# [standard, l_shape]
parser.add_argument('--needle', type=str, default='standard')
parser.add_argument('--arm', type=str, default='psm2') # [psm1, psm2]
parser.add_argument('--preprocess-type', type=str, default='segment_script') # [segment_net, mixdepth,origin, segment_script]
parser.add_argument('--image-type', type=str, default='zoom_needle_gripper_boximage') #[zoom_needle_gripper_boximage, zoom_needle_boximage]
parser.add_argument('--segment-net-file', type=str, default="none")
parser.add_argument('--reset', type=str, default="manual") #["auto", "manual"]
parser.add_argument('--clutch', type=int, default=6)

args = parser.parse_args()

#==================================

section = args.section



if not args.only_eval:
  configs = yaml.safe_load((pathlib.Path("./examples/jsons/default_np.yaml")).read_text())
  config = common.Config(configs)
  if args.json !="":
    configs = yaml.safe_load((pathlib.Path(args.json)).read_text())
    config = config.update(configs)
    baseline = pathlib.Path(args.json).stem
  else:
    baseline = "DreamerBC"


  _env_name = args.robot+"-"+args.platform+"-"+args.arm+"-"+ args.preprocess_type+"-"+args.image_type+"-prefill"+str(args.prefill)+"-clutch"+str(args.clutch) 


  logdir = str(Path('./data/suture/needle_picking') / _env_name / baseline / str(section) )
  config = config.update({
    'bc_dir': logdir + '/train_episodes/oracle',
    'logdir': logdir,         
    'eval_eps': args.eval_eps,
    'seed': args.seed,
    'prefill' : args.prefill
                  })
  if args.timelimit >= 0:
    config = config.update({"time_limit": args.timelimit})
else:
  assert args.json!="", "please specify json file"
  configs = yaml.safe_load((pathlib.Path(args.json)).read_text())
  config = common.Config(configs)

if args.preprocess_type == "segment_net":
  assert args.segment_net_file!="none", "please specify a weight file for segmentation net"
env = make_env(
              robot_type=args.robot,
             platform_type=args.platform, #[cuboid, phantom]
              preprocess_type=args.preprocess_type, 
             image_type=args.image_type,
            #  scalar2image_obs_key=["gripper_state", "state"],
             action_arm_device=args.arm,
            reset_needle_mode=args.reset,
             clutch_start_engaged=args.clutch,
             resize_resolution=64,
             timelimit=config.time_limit, 
            segment_net_file=args.segment_net_file,
            #  is_depth=True, 
            #  is_idle_action=False,
             is_ds4_oracle=False,
             is_visualizer=False,
            #  is_visualizer_blocking=True, 
             is_dummy=config.is_pure_train,
            needle_type=args.needle,
)


env.seed = args.seed

if args.only_eval:
  assert env.seed != config.seed, f"Seed for training is {env.seed}, use other seed number for evaluation!"
  _path = Path(args.json).parent
  print(_path)
  dv2_eval.eval_agnt(env=env,
                config=config, 
                logdir=_path, 
                time_limit=args.timelimit if args.timelimit>0 else config.time_limit, 
                eval_eps=args.eval_eps)
else:
  dv2.train(env, config, time_limit=config.time_limit, is_pure_train=args.only_train, is_pure_datagen=args.only_datagen)

env.close()
