from gym_suture.envs.wrapper import make_env
import dreamerv2.api as dv2
from pathlib import Path
import ruamel.yaml as yaml
from dreamerv2 import common
import pathlib


section = 10
baseline = "DreamerBC"
image_preprocess_type = 'segment_script'
logdir = str(Path('./data/suture/needle_picking/ambf') / baseline / image_preprocess_type / str(section) )


#=========================================
in_json_path = pathlib.Path(__file__).parent / 'jsons'/ 'dreamer2suture_1.yaml'
configs = yaml.safe_load((in_json_path).read_text())
defaults = common.Config(configs)
config = defaults.update({
  'bc_dir': logdir + '/train_episodes/oracle',
  'logdir': logdir,                 
                
                 }).parse_flags()


print(config)

env = make_env('ambf_needle_picking_64x64_discrete',  
                     is_visualizer=False, 
             image_preprocess_type='segment_script', 
             is_depth=True, 
             is_gripper_state_image=True,
             is_idle_action=False,
             is_ds4_oracle=False,
             action_arm_device='psm2',
             obs_type="image",
             timelimit=None,
             resize_resolution=64)

dv2.train(env, config, is_pure_train=config.is_pure_train, is_pure_datagen=config.is_pure_datagen)

env.close()
