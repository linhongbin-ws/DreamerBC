import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

import common
import expl
import time

class Agent(common.Module):

  def __init__(self, config, obs_space, act_space, step, **kwargs):
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step
    self.tfstep = tf.Variable(int(self.step), tf.int64)
    self.wm = WorldModel(config, obs_space, self.tfstep)
    if config.actor_type == "ActorCritic":
      self._task_behavior = ActorCritic(config, self.act_space, self.tfstep)
    elif config.actor_type == "AlphaZero":
      from dreamerv2 import mcts
      self._task_behavior = mcts.AlphaZero(config, self.act_space, self.tfstep)
    else:
      raise NotImplementedError
    if config.expl_behavior == 'greedy':
      self._expl_behavior = self._task_behavior
    else:
      if config.expl_behavior == 'Oracle':
        self._expl_behavior = getattr(expl, config.expl_behavior)(
            self.config, self.act_space, self.wm, self.tfstep,
            lambda seq: self.wm.heads['reward'](seq['feat']).mode(), env=kwargs['env'])
      else:
        self._expl_behavior = getattr(expl, config.expl_behavior)(
            self.config, self.act_space, self.wm, self.tfstep,
            lambda seq: self.wm.heads['reward'](seq['feat']).mode())

  @tf.function
  def policy(self, obs, state=None, mode='train'):
    obs = tf.nest.map_structure(tf.tensor, obs)
    tf.py_function(lambda: self.tfstep.assign(
        int(self.step), read_value=False), [], [])
    if state is None:
      latent = self.wm.rssm.initial(len(obs['reward']))
      action = tf.zeros((len(obs['reward']),) + self.act_space.shape)
      state = latent, action
    latent, action = state
    embed = self.wm.encoder(self.wm.preprocess(obs))
    sample = (mode == 'train') or not self.config.eval_state_mean
    latent, _ = self.wm.rssm.obs_step(
        latent, action, embed, obs['is_first'], sample)
    feat= self.wm.rssm.get_feat(latent)
    if mode == 'eval':
      actor = self._task_behavior.actor(feat)
      action = actor.mode()
      noise = self.config.eval_noise
    elif mode == 'explore':
      actor = self._expl_behavior.actor(feat)
      action = actor.sample()
      noise = self.config.expl_noise
    elif mode == 'train':
      actor = self._task_behavior.actor(feat)
      action = actor.sample()
      noise = self.config.expl_noise
    action = common.action_noise(action, noise, self.act_space)
    outputs = {'action': action}
    state = (latent, action)
    return outputs, state

  @tf.function
  def train(self, data,  bc_data=None, state=None, force=False):
    metrics = {}
    if bc_data is None:
      _state = state if self.config.train_carrystate else None
    else:
      _state = state[0] if self.config.train_carrystate else None
      _bc_state = state[1] if self.config.train_carrystate else None
    state, outputs, mets = self.wm.train(data, _state)
    metrics.update(mets)
    
    if bc_data is not None:
      bc_state, bc_outputs, bc_mets = self.wm.train(bc_data, _bc_state)
      bc_mets = {'bc_retrain_' + k: v for k, v in bc_mets.items()}
      metrics.update(bc_mets)
    # if self.tfstep > self.config.train_only_wm_steps or force: 
    start = outputs['post']
    reward = lambda seq: self.wm.heads['reward'](seq['feat']).mode()
    
    metrics.update(self._task_behavior.train(
        self.wm, start, data['is_terminal'], reward, bc_data=bc_data, bc_state=_bc_state))
    if bc_data is not None and self.config.bc_data_agent_retrain:
      bc_behav_met = self._task_behavior.train(
          self.wm, bc_outputs['post'], bc_data['is_terminal'], reward, bc_data=bc_data, bc_state=_bc_state)
      metrics.update({'bc_retrain_'+ k: v for k,v in bc_behav_met.items()}  )
    
    if self.config.expl_behavior != 'greedy':
      mets = self._expl_behavior.train(start, outputs, data)[-1]
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    
    if bc_data is None:
      out_state = state
    else:
      out_state = (state, bc_state)
    return out_state, metrics

  @tf.function
  def report(self, data):
    report = {}
    data = self.wm.preprocess(data)
    for key in self.wm.heads['decoder'].cnn_keys:
      name = key.replace('/', '_')
      report[f'openl_{name}'] = self.wm.video_pred(data, key)
    return report
  
  def save_sep(self, dir):
    dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {str(dir)}")
    self.wm.save(dir / 'wm.pkl')
    self._task_behavior.save(dir / 'policy.pkl')
  
  def load_sep(self, dir):
    _file = dir / 'wm.pkl'
    if _file.exists():
      self.wm.load(_file)
      print(f"load {str(_file)}")
      
    _file = dir / 'policy.pkl'
    if _file.exists():
      self._task_behavior.load(_file)   
      print(f"load {str(_file)}")



class WorldModel(common.Module):

  def __init__(self, config, obs_space, tfstep):
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    self.config = config
    self.tfstep = tfstep
    self.rssm = common.EnsembleRSSM(**config.rssm)
    self.encoder = common.Encoder(shapes, **config.encoder)
    self.heads = {}
    self.heads['decoder'] = common.Decoder(shapes, **config.decoder)
    self.heads['reward'] = common.MLP([], **config.reward_head)
    if config.pred_discount:
      self.heads['discount'] = common.MLP([], **config.discount_head)
    for name in config.grad_heads:
      assert name in self.heads, name
    self.model_opt = common.Optimizer('model', **config.model_opt)

  def train(self, data, state=None):
    with tf.GradientTape() as model_tape:
      model_loss, state, outputs, metrics = self.loss(data, state)
    modules = [self.encoder, self.rssm, *self.heads.values()]
    metrics.update(self.model_opt(model_tape, model_loss, modules))
    return state, outputs, metrics

  def loss(self, data, state=None):
    data = self.preprocess(data)
    embed = self.encoder(data)
    post, prior = self.rssm.observe(
        embed, data['action'], data['is_first'], state)
    kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
    assert len(kl_loss.shape) == 0
    likes = {}
    losses = {'kl': kl_loss}
    feat = self.rssm.get_feat(post)
    for name, head in self.heads.items():
      grad_head = (name in self.config.grad_heads)
      inp = feat if grad_head else tf.stop_gradient(feat)
      out = head(inp)
      dists = out if isinstance(out, dict) else {name: out}
      for key, dist in dists.items():
        if key.find('image_c') == -1:
          if key.find('image') == -1 or self.config.grad_uniform_image:
            like = tf.cast(dist.log_prob(data[key]), tf.float32)
            likes[key] = like
            losses[key] = -like.mean()
        else:
          _i = int(key[7:])
          if self.config.grad_extra_image_channel_scale[_i] > 0:
            like = tf.cast(dist.log_prob(data['image'][:,:,:,:,_i]), tf.float32)\
                * self.config.grad_extra_image_channel_scale[_i] / sum(self.config.grad_extra_image_channel_scale) * len(self.config.grad_extra_image_channel_scale)
            likes[key] = like
            losses[key] = -like.mean()
            
    model_loss = sum(
        self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items())
    # if quant_loss is not None:
    #   model_loss += quant_loss
    outs = dict(
        embed=embed, feat=feat, post=post,
        prior=prior, likes=likes, kl=kl_value)
    metrics = {f'{name}_loss': value for name, value in losses.items()}
    # if quant_loss is not None:
    #   metrics['quant_loss'] = quant_loss
    metrics['model_kl'] = kl_value.mean()
    metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean()
    metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean()
    last_state = {k: v[:, -1] for k, v in post.items()}
    return model_loss, last_state, outs, metrics

  def imagine(self, policy, start, is_terminal, horizon, actor_type='ActorCritic'):
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}
    start['feat'] = self.rssm.get_feat(start)
    if actor_type == 'MCTS':
      action, action_prob = policy.get_action(start, wm=self, n_playout=1)
      start['action'] = tf.zeros_like(action.mode())
      start['action_prob'] = tf.zeros_like(action_prob)
      policy.update_tree(None)
      _start_whole_t = time.time()
    else:
      start['action'] = tf.zeros_like(policy(start['feat']).mode())
    seq = {k: [v] for k, v in start.items()}
    for _ in range(horizon):
      _in = {k:tf.stop_gradient(v[-1]) for k,v in seq.items()} if actor_type=='MCTS' else tf.stop_gradient(seq['feat'][-1])
      if actor_type == 'MCTS':
        # _start_t = time.time()
        action, action_prob = policy.get_action(start, wm=self, n_playout=None)
        action = action.sample()
        seq['action_prob'].append(action_prob)
        policy.update_tree(tf.math.argmax(action[0], axis=0).numpy())
        # print("elapse time: ",time.time() - _start_t)
      else:
        action = policy(_in).sample()
      
      state = self.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
      feat = self.rssm.get_feat(state)
      for key, value in {**state, 'action': action, 'feat': feat}.items():
        seq[key].append(value)

    seq = {k: tf.stack(v, 0) for k, v in seq.items()}
    if 'discount' in self.heads:
      disc = self.heads['discount'](seq['feat']).mean()
      if is_terminal is not None:
        # Override discount prediction for the first step with the true
        # discount factor from the replay buffer.
        true_first = 1.0 - flatten(is_terminal).astype(disc.dtype)
        true_first *= self.config.discount
        disc = tf.concat([true_first[None], disc[1:]], 0)
    else:
      disc = self.config.discount * tf.ones(seq['feat'].shape[:-1])
    seq['discount'] = disc
    # Shift discount factors because they imply whether the following state
    # will be valid, not whether the current state is valid.
    seq['weight'] = tf.math.cumprod(
        tf.concat([tf.ones_like(disc[:1]), disc[:-1]], 0), 0)
    if actor_type == 'MCTS':
      policy.update_tree(None)
      print("gen 1 traj, time: ",time.time() - _start_whole_t)
    return seq

  @tf.function
  def preprocess(self, obs):
    dtype = prec.global_policy().compute_dtype
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_'):
        continue
      if value.dtype == tf.int32:
        value = value.astype(dtype)
      if value.dtype == tf.uint8:
        value = value.astype(dtype) / 255.0 - 0.5
      obs[key] = value
    obs['reward'] = {
        'identity': tf.identity,
        'sign': tf.sign,
        'tanh': tf.tanh,
    }[self.config.clip_rewards](obs['reward'])
    obs['discount'] = 1.0 - obs['is_terminal'].astype(dtype)
    obs['discount'] *= self.config.discount
    return obs

  @tf.function
  def video_pred(self, data, key):
    decoder = self.heads['decoder']
    truth = data[key][:6] + 0.5
    embed = self.encoder(data)
    states, _ = self.rssm.observe(
        embed[:6, :5], data['action'][:6, :5], data['is_first'][:6, :5])
    recon = decoder(self.rssm.get_feat(states))[key].mode()[:6]
    init = {k: v[:, -1] for k, v in states.items()}
    prior = self.rssm.imagine(data['action'][:6, 5:], init)
    openl = decoder(self.rssm.get_feat(prior))[key].mode()
    model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2
    video = tf.concat([truth, model, error], 2)
    B, T, H, W, C = video.shape
    return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))


class ActorCritic(common.Module):

  def __init__(self, config, act_space, tfstep):
    self.config = config
    self.act_space = act_space
    self.tfstep = tfstep
    discrete = hasattr(act_space, 'n')
    if self.config.actor.dist == 'auto':
      self.config = self.config.update({
          'actor.dist': 'onehot' if discrete else 'trunc_normal'})
    if self.config.actor_grad == 'auto':
      self.config = self.config.update({
          'actor_grad': 'reinforce' if discrete else 'dynamics'})
    self.actor = common.MLP(act_space.shape[0], **self.config.actor)
    self.critic = common.MLP([], **self.config.critic)
    if self.config.slow_target:
      self._target_critic = common.MLP([], **self.config.critic)
      self._updates = tf.Variable(0, tf.int64)
    else:
      self._target_critic = self.critic
    self.actor_opt = common.Optimizer('actor', **self.config.actor_opt)
    self.critic_opt = common.Optimizer('critic', **self.config.critic_opt)
    if self.config.reward_norm_skip:
      self.rewnorm = None
    else:
      self.rewnorm = common.StreamNorm(**self.config.reward_norm)

  def train(self, world_model, start, is_terminal, reward_fn, bc_data=None, bc_state=None,**kwargs):
    metrics = {}
    hor = self.config.imag_horizon
    # The weights are is_terminal flags for the imagination start states.
    # Technically, they should multiply the losses from the second trajectory
    # step onwards, which is the first imagined step. However, we are not
    # training the action that led into the first step anyway, so we can use
    # them to scale the whole sequence.
    with tf.GradientTape() as actor_tape:
      seq = world_model.imagine(self.actor, start, is_terminal, hor)
      reward = reward_fn(seq)
      if self.rewnorm is not None:
        seq['reward'], mets1 = self.rewnorm(reward)
        mets1 = {f'reward_{k}': v for k, v in mets1.items()}
        metrics.update(**mets1)
      else:
         seq['reward'] = reward
      target, mets2 = self.target(seq)
      # if bc_data is None:
      actor_loss, mets3 = self.actor_loss(seq, target)
      mets3['actor_pure_loss'] = actor_loss
      # else:
      if bc_data is not None and self.config.bc_loss:
        data = world_model.preprocess(bc_data)
        embed = world_model.encoder(data)
        post, prior = world_model.rssm.observe(embed, data['action'], data['is_first'], state=bc_state)
        feat = world_model.rssm.get_feat(post)
        action = self.actor(tf.stop_gradient(feat[:,self.config.bc_skip_start_step_num:-1,:])) # action is prev action, needs to shift 1 
        bc_grad_weight = common.schedule(self.config.bc_grad_weight, self.tfstep)
        like = -tf.cast(action.log_prob(data['action'][:,1+self.config.bc_skip_start_step_num:,:]), tf.float32).mean() 
        actor_loss = like * bc_grad_weight + actor_loss 
        mets3['actor_bc_loss'] = like
        mets3['bc_grad_weight'] = bc_grad_weight
        
    with tf.GradientTape() as critic_tape:
      critic_loss, mets4 = self.critic_loss(seq, target)
    metrics.update(self.actor_opt(actor_tape, actor_loss, self.actor))
    metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))
    metrics.update(**mets2, **mets3, **mets4)
    self.update_slow_target()  # Variables exist after first forward pass.
    return metrics

  def actor_loss(self, seq, target):
    # Actions:      0   [a1]  [a2]   a3
    #                  ^  |  ^  |  ^  |
    #                 /   v /   v /   v
    # States:     [z0]->[z1]-> z2 -> z3
    # Targets:     t0   [t1]  [t2]
    # Baselines:  [v0]  [v1]   v2    v3
    # Entropies:        [e1]  [e2]
    # Weights:    [ 1]  [w1]   w2    w3
    # Loss:              l1    l2
    metrics = {}
    # Two states are lost at the end of the trajectory, one for the boostrap
    # value prediction and one because the corresponding action does not lead
    # anywhere anymore. One target is lost at the start of the trajectory
    # because the initial state comes from the replay buffer.
    policy = self.actor(tf.stop_gradient(seq['feat'][:-2]))
    if self.config.actor_grad == 'dynamics':
      objective = target[1:]
    elif self.config.actor_grad == 'reinforce':
      baseline = self._target_critic(seq['feat'][:-2]).mode()
      advantage = tf.stop_gradient(target[1:] - baseline)
      action = tf.stop_gradient(seq['action'][1:-1])
      objective = policy.log_prob(action) * advantage
    elif self.config.actor_grad == 'both':
      baseline = self._target_critic(seq['feat'][:-2]).mode()
      advantage = tf.stop_gradient(target[1:] - baseline)
      objective = policy.log_prob(seq['action'][1:-1]) * advantage
      mix = common.schedule(self.config.actor_grad_mix, self.tfstep)
      objective = mix * target[1:] + (1 - mix) * objective
      metrics['actor_grad_mix'] = mix
    else:
      raise NotImplementedError(self.config.actor_grad)
    ent = policy.entropy()
    ent_scale = common.schedule(self.config.actor_ent, self.tfstep)
    objective += ent_scale * ent
    weight = tf.stop_gradient(seq['weight'])
    actor_loss = -(weight[:-2] * objective).mean()
    metrics['actor_ent'] = ent.mean()
    metrics['actor_ent_scale'] = ent_scale
    return actor_loss, metrics

  def critic_loss(self, seq, target):
    # States:     [z0]  [z1]  [z2]   z3
    # Rewards:    [r0]  [r1]  [r2]   r3
    # Values:     [v0]  [v1]  [v2]   v3
    # Weights:    [ 1]  [w1]  [w2]   w3
    # Targets:    [t0]  [t1]  [t2]
    # Loss:        l0    l1    l2
    dist = self.critic(seq['feat'][:-1])
    target = tf.stop_gradient(target)
    weight = tf.stop_gradient(seq['weight'])
    critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()
    metrics = {'critic': dist.mode().mean()}
    return critic_loss, metrics

  def target(self, seq):
    # States:     [z0]  [z1]  [z2]  [z3]
    # Rewards:    [r0]  [r1]  [r2]   r3
    # Values:     [v0]  [v1]  [v2]  [v3]
    # Discount:   [d0]  [d1]  [d2]   d3
    # Targets:     t0    t1    t2
    reward = tf.cast(seq['reward'], tf.float32)
    disc = tf.cast(seq['discount'], tf.float32)
    value = self._target_critic(seq['feat']).mode()
    # Skipping last time step because it is used for bootstrapping.
    target = common.lambda_return(
        reward[:-1], value[:-1], disc[:-1],
        bootstrap=value[-1],
        lambda_=self.config.discount_lambda,
        axis=0)
    metrics = {}
    metrics['critic_slow'] = value.mean()
    metrics['critic_target'] = target.mean()
    return target, metrics

  def update_slow_target(self):
    if self.config.slow_target:
      if self._updates % self.config.slow_target_update == 0:
        mix = 1.0 if self._updates == 0 else float(
            self.config.slow_target_fraction)
        for s, d in zip(self.critic.variables, self._target_critic.variables):
          d.assign(mix * s + (1 - mix) * d)
        tf.print("[info] slow critic update!")
      self._updates.assign_add(1)
