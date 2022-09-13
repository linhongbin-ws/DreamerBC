import gym
import gym_minigrid
import dreamerv2.api as dv2
from gym.spaces import Dict

config = dv2.defaults.update({
    'logdir': './data/minigrid/MCTS',
    'log_every': 1e3,
    'train_every': 10,
    'prefill': 1e4,
    'actor_ent': 3e-3,
    'loss_scales.kl': 1.0,
    'discount': 0.99,
}).parse_flags()

env = gym.make('MiniGrid-DoorKey-6x6-v0')
env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)
env.observation_space = Dict({k:v for k,v in env.observation_space.items() if k != 'mission'})
dv2.train(env, config, is_train=config.istrain)
