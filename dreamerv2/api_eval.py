import collections
import logging
import os
import pathlib
import re
import sys
import warnings
import tracemalloc
import linecache

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml

import agent
import common

from common import Config
from common import GymWrapper
from common import RenderImage
from common import TerminalOutput
from common import JSONLOutput
from common import TensorBoardOutput
from tqdm import tqdm

configs = yaml.safe_load(
    (pathlib.Path(__file__).parent / 'configs.yaml').read_text())
defaults = common.Config(configs.pop('defaults'))

import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

    
def eval_agnt(env, config, logdir, time_limit=-1, eval_eps=50):
  tf.config.experimental_run_functions_eagerly(not config.jit)
  logdir = pathlib.Path(logdir).expanduser()
  evaldir = logdir / 'eval_result'
  evaldir.mkdir(parents=True, exist_ok=True)

  step = common.Counter(0)
  outputs =[
      common.TerminalOutput(),
      common.JSONLOutput(str(logdir)),
      common.TensorBoardOutput(str(logdir)),
  ]

  logger = common.Logger(step, outputs, multiplier=config.action_repeat)
  metrics = collections.defaultdict(list)

  env = common.GymWrapper(env)
  env = common.ResizeImage(env)
  if hasattr(env.act_space['action'], 'n'):
    env = common.OneHotAction(env)
  else:
    env = common.NormalizeAction(env)
  if time_limit>0:
    env = common.TimeLimit(env, time_limit)

   
  eval_replay = common.Replay(evaldir / 'eval_episodes', **dict(
    capacity=config.replay.capacity // 10,
    minlen=config.dataset.length,
    maxlen=config.dataset.length))

  
    
  eval_driver = common.Driver([env])
  # eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
  # eval_driver.on_episode(eval_replay.add_episode)


  print('Create agent..')
  agnt = agent.Agent(config, env.obs_space, env.act_space, step, env=env)
  # print(f"model load path: {str(logdir)}")
  # agnt.load_sep(logdir)
  eval_policy = lambda *args: agnt.policy(*args, mode='eval')

  train_replay = common.Replay(logdir / 'train_mini_data', **config.replay)
  train_driver = common.Driver([env])
  train_driver.on_step(train_replay.add_step)
  train_driver.on_reset(train_replay.add_step)
  while True:
    try:
      train_dataset = iter(train_replay.dataset(**config.dataset))
      next(train_dataset)
    except Exception as e:
      print("no mini data")
      print("fill 1 eps training eps...")
      random_agnt = common.RandomAgent(env.act_space)
      train_driver(random_agnt, episodes=1)
    else:
      break
  bc_func = lambda dataset: next(dataset) if dataset is not None else None
  train_agent = common.CarryOverState(agnt.train, is_bc=True)
  train_agent(next(train_dataset), next(train_dataset))

  agnt.load_sep(logdir)

  eval_stat = {'average_scores':0, 
              'sucess_eps_count':0, 
              'sucess_eps_rate':0, 
              'eps_cnt':0, 
              'filter_cases_cnt': 0, 
              'filter_state_4_cnt':0, 
              'filter_state_5_cnt':0}
  # eval_stat = {'average_scores':0, 'sucess_eps_count':0, 'sucess_eps_rate':0, 'eps_cnt':0, 'filter_cases_cnt': 0, 'filter_state_4_cnt':0, 'filter_state_5_cnt':0}
  def eval_sucess_count(ep):
    score = float(ep['reward'].astype(np.float64).sum())
    eval_stat['eps_cnt'] +=1
    eval_stat['average_scores'] += score
    print(ep['state'][-1][0]==2.0)
    print("states:",ep['state'])
    if ep['state'][-1][0] == 1.0:
      eval_stat['sucess_eps_count'] += 1
    if ep['state'][-1][0] >=3.0:
      eval_stat['filter_cases_cnt'] +=1
      print(f"Bad filter case {ep['state'][-1]}!")
      _str = f"filter_state_{ep['state'][-1]}_cnt"
      if not _str in eval_stat:
        eval_stat[_str]=1
      else:
        eval_stat[_str] +=1
    print(f"sucess/total/filter_cases/goal: ({eval_stat['sucess_eps_count']}/ {eval_stat['eps_cnt']} / {eval_stat['filter_cases_cnt']}) / {eval_eps}")
  eval_driver.on_episode(eval_sucess_count)


  print("===========================================")
  print("Evaluate phase")
  eval_driver.reset()
  for k,v in eval_stat.items():
    eval_stat[k] = 0

  while (eval_stat['eps_cnt'] - eval_stat['filter_cases_cnt']) < eval_eps and eval_stat['eps_cnt'] < eval_eps:
    eval_driver(eval_policy, episodes=1)

  eval_stat['average_scores'] = eval_stat['average_scores'] / \
      eval_stat['eps_cnt']
  eval_stat['sucess_eps_rate'] = eval_stat['sucess_eps_count']/eval_stat['eps_cnt']
  eval_stat['sucess_eps_filter_rate'] = eval_stat['sucess_eps_count'] / \
      (eval_stat['eps_cnt'] - eval_stat['filter_cases_cnt']
        ) if (eval_stat['eps_cnt'] - eval_stat['filter_cases_cnt']) >= 1 else 0
  logger.add(eval_stat, prefix='eval')
  if should_video_eval(step):
    logger.add(agnt.report(next(eval_dataset)), prefix='eval')
  print("==============")
  print(f"# eval rate: {eval_stat['sucess_eps_rate']} !!")
  print(f"# eval filter rate: {eval_stat['sucess_eps_filter_rate']} !!")
  print(f"# eval average return: {eval_stat['average_scores']}")
  logger.write(fps=True)
