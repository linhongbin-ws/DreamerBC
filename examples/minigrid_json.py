import gym
import gym_minigrid
from dreamerv2 import common
import dreamerv2.api_other as dv2
from gym.spaces import Dict
import argparse
import ruamel.yaml as yaml
import pathlib
parser = argparse.ArgumentParser()
parser.add_argument('--json', type=str, default="./examples/jsons/minigrid_config.yaml")
parser.add_argument('--section', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

configs = yaml.safe_load((pathlib.Path(args.json)).read_text())
config = common.Config(configs)

logdir = str(pathlib.Path('./data/minigrid') /str(args.section) )
config = config.update({
'bc_dir': '',
'logdir': logdir,         
'seed': args.seed,
                  })

env = gym.make('MiniGrid-DoorKey-6x6-v0')
env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)
env.observation_space = Dict({k:v for k,v in env.observation_space.items() if k != 'mission'})
dv2.train(env, config,time_limit=config.time_limit)
