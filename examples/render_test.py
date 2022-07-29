from dreamerv2 import common
from dreamerv2.common import Config
from dreamerv2.common import GymWrapper
from dreamerv2.common import RenderImage
from dreamerv2.common import TerminalOutput
from dreamerv2.common import JSONLOutput
from dreamerv2.common import TensorBoardOutput
from gym_suture.envs.wrapper import make_env 

is_visualize = True
env = make_env('ambf_needle_picking_64x64_discrete',is_visualizer=is_visualize)
env = common.GymWrapper(env)
env = common.ResizeImage(env)
# if hasattr(env.act_space['action'], 'n'):
#     env = common.OneHotAction(env)
# else:
#     env = common.NormalizeAction(env)
env = common.TimeLimit(env, 100)





env.seed = 34



env.reset()
cnt =0 
done =False
while not done:
    # action = env.action_space.sample()
    # action = env.get_oracle_action()
    action = {'action':env.action_space.sample()}
    print(action)
    # action = 0
    # print("oracle:", action)
    # action = env.get_oracle_action(noise_scale=0.15)
    # action = env.get_oracle_action(noise_scale=0.0)
    obs= env.step(action)
    # print("reward:", reward,"action", action, "info", info, "cnt", cnt)
    # print("info:", info)
    cnt+=1
    
    if is_visualize:
        env.render()
        if not env.is_active:
            break
        
    # print(f'reward:{reward} steps: {cnt}')
        
    # spin()
env.close()
