import gym_minigrid
import gym
from time import sleep
env = gym.make('MiniGrid-DoorKey-6x6-v0')
# env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)
env.reset()
env.render(mode='human')
sleep(5)