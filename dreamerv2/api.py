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

# def display_top(snapshot, key_type='lineno', limit=3):
#   snapshot = snapshot.filter_traces((
#       tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
#       tracemalloc.Filter(False, "<unknown>"),
#   ))
#   top_stats = snapshot.statistics(key_type)

#   print("Top %s lines" % limit)
#   for index, stat in enumerate(top_stats[:limit], 1):
#       frame = stat.traceback[0]
#       # replace "/path/to/module/file.py" with "module/file.py"
#       filename = os.sep.join(frame.filename.split(os.sep)[-2:])
#       print("#%s: %s:%s: %.1f KiB"
#             % (index, filename, frame.lineno, stat.size / 1024))
#       line = linecache.getline(frame.filename, frame.lineno).strip()
#       if line:
#           print('    %s' % line)

#   other = top_stats[limit:]
#   if other:
#       size = sum(stat.size for stat in other)
#       print("%s other: %.1f KiB" % (len(other), size / 1024))
#   total = sum(stat.size for stat in top_stats)
#   print("Total allocated size: %.1f KiB" % (total / 1024))
    
def train(env, config, outputs=None, is_pure_train=False, is_pure_datagen=False, skip_gym_wrap=False):
  assert not (is_pure_train and is_pure_datagen)
  tf.config.experimental_run_functions_eagerly(not config.jit)
  logdir = pathlib.Path(config.logdir).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)
  offlinelogdir = logdir / 'offline' if is_pure_train else logdir
  offlinelogdir.mkdir(parents=True, exist_ok=True)
  config.save(logdir / 'config.yaml')
  print(config, '\n')
  print('Logdir', logdir)

  train_replay = common.Replay(logdir / 'train_episodes', **config.replay)
  eval_replay = common.Replay(logdir / 'eval_episodes', **dict(
    capacity=config.replay.capacity // 10,
    minlen=config.dataset.length,
    maxlen=config.dataset.length))
  step = common.Counter(train_replay.stats['total_steps'])
  outputs = outputs or [
      common.TerminalOutput(),
      common.JSONLOutput(str(offlinelogdir)),
      common.TensorBoardOutput(str(offlinelogdir)),
  ]

  logger = common.Logger(step, outputs, multiplier=config.action_repeat)
  metrics = collections.defaultdict(list)

  should_train = common.Every(config.train_every)
  should_log = common.Every(config.log_every)
  should_video_train = common.Every(config.eval_every)
  should_video_eval = common.Every(config.eval_every)
  should_expl = common.Until(config.expl_until)

  step = common.Counter(0) 


  if not skip_gym_wrap:
    env = common.GymWrapper(env)
    env = common.ResizeImage(env)
    if hasattr(env.act_space['action'], 'n'):
      env = common.OneHotAction(env)
    else:
      env = common.NormalizeAction(env)
    env = common.TimeLimit(env, config.time_limit)

  if not is_pure_train:
    
    # prefill agent
    prefill_replay = common.Replay(logdir / 'train_episodes' / config.prefill_agent, **config.replay)
    prefill_driver = common.Driver([env])
    prefill_driver.on_step(lambda tran, worker: step.increment())
    prefill_driver.on_step(prefill_replay.add_step)
    prefill_driver.on_reset(prefill_replay.add_step)

    prefill = max(0, config.prefill - prefill_replay.stats['total_steps'])
    if prefill:
      print(f'Prefill dataset ({prefill} steps).')
      if config.prefill_agent == 'random':
        prefill_agent = common.RandomAgent(env.act_space)
      elif config.prefill_agent == 'oracle':
        prefill_agent = common.OracleAgent(env.act_space, env=env)
      prefill_driver(prefill_agent, steps=prefill, episodes=1)
      prefill_driver.reset()
    
      del prefill_replay
      del prefill_driver
      
      train_replay = common.Replay(logdir / 'train_episodes', **config.replay)
      eval_replay = common.Replay(logdir / 'eval_episodes', **dict(
        capacity=config.replay.capacity // 10,
        minlen=config.dataset.length,
        maxlen=config.dataset.length))

      
    step = common.Counter(train_replay.stats['total_steps'])
    logger = common.Logger(step, outputs, multiplier=config.action_repeat)
    def per_episode(ep, mode):
      length = len(ep['reward']) - 1
      score = float(ep['reward'].astype(np.float64).sum())
      print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
      logger.scalar(f'{mode}_return', score)
      logger.scalar(f'{mode}_length', length)
      for key, value in ep.items():
        if re.match(config.log_keys_sum, key):
          logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
        if re.match(config.log_keys_mean, key):
          logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
        if re.match(config.log_keys_max, key):
          logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())
      should = {'train': should_video_train, 'eval': should_video_eval}[mode]
      if should(step):
        for key in config.log_keys_video:
          logger.video(f'{mode}_policy_{key}', ep[key])
      replay = dict(train=train_replay, eval=eval_replay)[mode]
      logger.add(replay.stats, prefix=mode)
      logger.write()
  
    # agent

    train_driver = common.Driver([env])
    train_driver.on_episode(lambda ep: per_episode(ep, mode='train'))
    train_driver.on_step(lambda tran, worker: step.increment())
    train_driver.on_step(train_replay.add_step)
    train_driver.on_reset(train_replay.add_step)
    
    eval_driver = common.Driver([env])
    eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
    eval_driver.on_episode(eval_replay.add_episode)
    
    prefill_eval_agent = common.RandomAgent(env.act_space)
    eval_driver(prefill_eval_agent, episodes=1)

  print('Create agent.')
  agnt = agent.Agent(config, env.obs_space, env.act_space, step, env=env)
  train_dataset = iter(train_replay.dataset(**config.dataset))
  eval_dataset = iter(eval_replay.dataset(**config.dataset))
  if config.bc_dir is not '':
    print(config.bc_dir)
    bc_dir = pathlib.Path(config.bc_dir)
    bc_replay = common.Replay(bc_dir, **config.replay)
    bc_dataset = iter(bc_replay.dataset(**config.dataset))
  else:
    bc_dataset = None
  bc_func = lambda dataset: next(dataset) if dataset is not None else None
  train_agent = common.CarryOverState(agnt.train)
  train_agent(next(train_dataset), bc_func(bc_dataset))
  agnt.load_sep(logdir)
  train_policy = lambda *args: agnt.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  eval_policy = lambda *args: agnt.policy(*args, mode='eval')

  if is_pure_train:
    pbar = tqdm(range(config.offline_step))
    for _s in pbar:
      step.increment()
      tf.py_function(lambda: agnt.tfstep.assign(
        int(step), read_value=False), [], [])
      mets = train_agent(next(train_dataset), bc_func(bc_dataset))
      des_str = f"actor_pure: {mets['actor_pure_loss'].numpy()} critic: {mets['critic_loss'].numpy()}"
      if bc_dataset is not None:
        des_str = des_str + f"bc: {mets['actor_bc_loss'].numpy()} "
      pbar.set_description(des_str)
      # _ = agnt.report(next(dataset))
      [metrics[key].append(value) for key, value in mets.items()]
      if should_log(step):
        for name, values in metrics.items():
          logger.scalar(name, np.array(values, np.float64).mean())
          metrics[name].clear()
        logger.add(agnt.report(next(train_dataset)))
        if bc_dataset is not None:
          bc_report = agnt.report(next(bc_dataset))
          logger.add({'bc_report_'+k: v for k,v in bc_report.items()})
        logger.write(fps=True)
        # agnt.save(logdir / 'variables.pkl')
        agnt.save_sep(logdir)
        print("save param")
        
        
        # ## facing serious memory leak, ref to https://github.com/tensorflow/tensorflow/issues/37653
        # _update = False
        # import psutil

        # # used_mem = psutil.virtual_memory().used
        # # print("used memory: {} Mb".format(used_mem / 1024 / 1024))
        # del dataset
        # replay.cleanup()
        # replay = common.Replay(logdir / 'train_episodes', **config.replay)
        # dataset = iter(replay.dataset(**config.dataset))
        # # snapshot = tracemalloc.take_snapshot()
        # # display_top(snapshot)

        
  else:
    if not is_pure_datagen:
      def train_step(tran, worker):
        # print(step.value)
        if should_train(step):
          pbar = tqdm(range(config.train_steps))
          for _ in pbar:
            mets = train_agent(next(train_dataset), bc_func(bc_dataset))
            [metrics[key].append(value) for key, value in mets.items()]
            des_str = f"actor_pure: {mets['actor_pure_loss'].numpy()} critic: {mets['critic_loss'].numpy()}"
            if bc_dataset is not None:
              des_str = des_str + f"bc: {mets['actor_bc_loss'].numpy()} "
            pbar.set_description(des_str)
        if should_log(step):
          for name, values in metrics.items():
            logger.scalar(name, np.array(values, np.float64).mean())
            metrics[name].clear()
            logger.add(agnt.report(next(train_dataset)), prefix='train')
          if bc_dataset is not None:
            bc_report = agnt.report(next(bc_dataset))
            logger.add({'bc_report_'+k: v for k,v in bc_report.items()} , prefix='train')
          logger.write(fps=True)
          agnt.save_sep(logdir)
          print("save param")
      train_driver.on_step(train_step)
      
      eval_stat = {'average_scores':0, 'sucess_eps_count':0, 'sucess_eps_rate':0}
      def eval_sucess_count(ep):
        score = float(ep['reward'].astype(np.float64).sum())
        eval_stat['average_scores'] += score
        eval_stat['sucess_eps_count'] += int(score>0) 
      eval_driver.on_episode(eval_sucess_count)
    while step < config.steps:

      
      print("===========================================")
      print("Evaluate phase")
      eval_driver.reset()
      for k,v in eval_stat.items():
        eval_stat[k] = 0
      eval_driver(eval_policy, episodes=config.eval_eps)
      eval_stat['average_scores'] = eval_stat['average_scores']/config.eval_eps
      eval_stat['sucess_eps_rate'] = eval_stat['sucess_eps_count']/config.eval_eps
      logger.add(eval_stat, prefix='eval')
      logger.add(agnt.report(next(eval_dataset)), prefix='eval')
      if eval_stat['sucess_eps_rate'] >= config.save_sucess_eps_rate:
        _model_dir = str(step.value) + '_'+str(int(eval_stat['sucess_eps_rate']*100))+'_percent'
        agnt.save_sep(logdir / 'model' / _model_dir)
        config.save(logdir / 'model' / _model_dir / 'config.yaml')
        print("save  eval param")

      print("===========================================")
      print("Train phase...")
      train_driver.reset()
      train_driver(train_policy, steps=config.eval_every)
      

      logger.write(fps=True)
      if is_pure_datagen:
        print("reload param")
        agnt.load_sep(logdir)

