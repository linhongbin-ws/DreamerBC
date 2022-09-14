from tkinter import NONE
import numpy as np
import common
import tensorflow as tf
from collections import deque
import random

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
    def __init__(self, parent, prior_p, state, discount, reward):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
        self._state = state
        self._discount = discount
        self._reward = reward

    def expand(self, nxt_s):
        for action, action_prob, nxt_state, discount, reward in nxt_s:
            if action not in self._children:
                self._children[action] = TreeNode(self, action_prob, nxt_state, discount, reward)

    def select(self, c_puct):
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(leaf_value*self._discount+self._reward)
        self.update(leaf_value)

    def get_value(self, c_puct):
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    def __init__(self, value_func, actor_func, action_n, c_puct=5, n_playout=10, terminate_discount=0.2):
        self._root = None
        self._value_func = value_func
        self._actor_func =  actor_func
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._action_n =  action_n
        self._terminate_discount = terminate_discount

    def _playout(self, wm):
        node = self._root
        while(1):
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)  # Greedily select next move.

        
        
        if node._discount >= self._terminate_discount:
            leaf_value = self._value_func(node._state['feat']).mode().numpy()
            actions, action_probs, nxt_states, discounts, rewards = self.wm_step(node._state, wm)
            nxt_s = zip(actions, action_probs, nxt_states, discounts, rewards)
            node.expand(nxt_s)
        else:
            leaf_value = node._reward
        node.update_recursive(leaf_value)

    def get_move_probs(self, state, wm, temp=1e-3, n_playout=None):
        if self._root is None:
            self._root = TreeNode(None, 1.0, state, 1, 0) 

        _n_playout = n_playout or self._n_playout
        # for n in range(_n_playout):
        while(self._root._n_visits<_n_playout):
            self._playout(wm)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        if len(act_visits) == 0:
            act_visits = [(i, 1) for i in range(self._action_n)]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = None 

    def get_action(self, state, wm, temp=1e-3, n_playout=None):
        assert state['deter'].shape[0] == 1 # only support batch == 1
        _, probs = self.get_move_probs(state, wm, temp, n_playout)
        # if is_train:
        #     move = np.random.choice(
        #         acts,
        #         p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
        #     )
        # else:
        #     move = np.random.choice(acts, p=probs)
        
        # self._root = None
        
        # if is_reset_tree:
        #     self.mcts.update_with_move(-1)
        # else:
        #     self.mcts.update_with_move(move)
        out_probs = np.array([probs])
        return common.OneHotDist(out_probs), tf.convert_to_tensor(out_probs, dtype=tf.float32)
    
    def update_tree(self, move=None):
        if move is None:
            self.update_with_move(-1)
        else:
            self.update_with_move(move)

    
    def wm_step(self, state, wm):
        _state = {k:tf.concat([v]*self._action_n, axis=0) for k,v in state.items()}
        _action = tf.one_hot([i for i in range(self._action_n)], depth=self._action_n)
        # start['feat'] = wm.rssm.get_feat(start)
        nxt_state = wm.rssm.img_step(_state, _action)
        nxt_state['feat'] = wm.rssm.get_feat(nxt_state)
        discount = wm.heads['discount'](nxt_state['feat']).mean()
        reward = wm.heads['reward'](nxt_state['feat']).mode()
        _action_probs = self._actor_func(state['feat'])
        # nxt_state['feat'] = feat
        actions = [i for i in range(self._action_n)]
        _action_probs = _action_probs.prob(_action).numpy()
        action_probs = _action_probs.tolist()
        nxt_states = [{k:v[i:i+1] for k,v in nxt_state.items()} for i in range(self._action_n)]
        discounts = [discount[i] for i in range(self._action_n)]
        rewards = [reward[i] for i in range(self._action_n)]

        return actions, action_probs, nxt_states, discounts, rewards

    
class AlphaZero(common.Module):
    def __init__(self, config, act_space, tfstep):
        self.config = config
        self.act_space = act_space
        self.tfstep = tfstep
        assert hasattr(act_space, 'n') # only support discrete
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
        self.rewnorm = common.StreamNorm(**self.config.reward_norm)
        
        self.mcts_planner = MCTS(value_func=self.critic, 
                                 actor_func=self.actor, 
                                 action_n=act_space.n, 
                                 c_puct=config.train_mcts_c_puct, 
                                 n_playout=config.train_mcts_n_playout)
        
        self.train_seq_buffer = deque(maxlen=self.config.train_seq_buffer)
        
        self.should_gen_traj = common.Every(config.train_mcts_gen_traj_every)
        
    def train(self, world_model, start, is_terminal, reward_fn, bc_data, **kwargs):
        metrics = {}
        if self.should_gen_traj(self.tfstep):
            hor = self.config.imag_horizon
            # The weights are is_terminal flags for the imagination start states.
            # Technically, they should multiply the losses from the second trajectory
            # step onwards, which is the first imagined step. However, we are not
            # training the action that led into the first step anyway, so we can use
            # them to scale the whole sequence.
            
            # _policy = lambda *args: self.mcts_planner.get_action(*args, wm=world_model, temp=1e-3, is_train=True)
            _policy = self.mcts_planner
            # seqs = None
            # targets = []
            i = random.randint(0, is_terminal.shape[0]-1)
            j = random.randint(0, is_terminal.shape[1]-1)
            # for i in range(1):
            #     for j in range(1):
            # for i in range(is_terminal.shape[0]):
            #     for j in range(is_terminal.shape[1]):
            
            _start = {k: v[i:i+1,j:j+1,...] for k, v in start.items()}
            _is_terminal = is_terminal[i:i+1,j:j+1]
            seq = world_model.imagine(_policy, _start, _is_terminal, hor, actor_type='MCTS')
            reward = reward_fn(seq)
            seq['reward'], mets1 = self.rewnorm(reward)
            mets1 = {f'reward_{k}': v for k, v in mets1.items()}
            
            target, mets2 = self.target(seq)
                    # if seqs is None:
                    #     seqs = {k:[v] for k, v in seq.items()}
                    # else:
                    #     seqs = {k:seqs[k]+[v] for k, v in seq.items()}
                    # targets.append(target)
            # seq = {k:tf.concat(v, 1)for k, v in seqs.items()}
            # target = tf.concat(targets, 1)
            self.train_seq_buffer.append((seq, target))
            print(f"traj buffer size:{len(self.train_seq_buffer)}")
            metrics.update(**mets1, **mets2)
            
        # bz = is_terminal.shape[0] * is_terminal.shape[1]
        bz = self.config.train_mcts_batch_size
        if len(self.train_seq_buffer) >= bz:
            mini_batch = random.sample(self.train_seq_buffer, bz)
            seqs = [data[0] for data in mini_batch]
            targets = [data[1] for data in mini_batch]
            seq = {k:tf.concat([i[k] for i in seqs], 1) for k in seqs[0].keys()}
            target = tf.concat(targets, 1)
            with tf.GradientTape() as critic_tape:
                critic_loss, mets4 = self.critic_loss(seq, target)
            metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))
            with tf.GradientTape() as actor_tape:
                actor_loss, mets3 = self.actor_loss(seq)
            metrics.update(self.actor_opt(actor_tape, actor_loss, self.actor))
            metrics.update(**mets3, **mets4)
            self.update_slow_target()  # Variables exist after first forward pass.
        return metrics



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
    
    def actor_loss(self, seq):
        # supervised learning actor
        metrics = {}
        policy = self.actor(tf.stop_gradient(seq['feat'][:-1]))

        objective = policy.log_prob(seq['action_prob'][1:])

        # ent = policy.entropy()
        ent_scale = common.schedule(self.config.actor_ent, self.tfstep)
        # objective += ent_scale * ent
        # weight = tf.stop_gradient(seq['weight'])
        # actor_loss = -(weight[:-2] * objective).mean()
        actor_loss = -objective.mean()
        # metrics['actor_ent'] = ent.mean()
        # metrics['actor_ent_scale'] = ent_scale
        return actor_loss, metrics

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
            self._updates.assign_add(1)