import tensorflow as tf
from tensorflow.keras import mixed_precision as prec
import common
import numpy as np
from tensorflow_probability import distributions as tfd

class MixPolicy(object):
    def __init__(self, mix, mean, std, prior_policy, amount):
        self.mix = mix
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        self.mean = flatten(tf.stack([mean]*amount, axis=0))
        self.std = flatten(tf.stack([std]*amount, axis=0))
        self.prior_policy = prior_policy
        self.cnt = -1
    def __call__(self, features):
        self.cnt+=1
        if np.random.uniform() > self.mix:
            _mean = self.mean[:,self.cnt,:]
            _std = self.std[:,self.cnt,:]
            dist = tfd.Normal(_mean, 
                            _std,
                            )
            return dist
        else:
            return self.prior_policy(features)
     


class CEM(object):
    def __init__(self,
                act_space,
                horizon=16,
                amount=10,
                topk=3,
                iteration=10,
                prior_mix=0.5,
                batch=2,
                loss_horizon=1,
                loss_scale=0.1,
                ):
        self.act_space = act_space
        self.horizon = horizon
        self.amount = amount
        self.topk=topk
        self.iteration = iteration
        self.prior_mix = prior_mix
        self.batch = batch
        self.loss_horizon = loss_horizon
        self.loss_scale = loss_scale

    def plan(self, 
            world_model, 
            actor_model, 
            start_state, 
            start_state_is_terminal, 
            target_func,
            reward_fn,
            rewnorm,
                ):
        
        amount, batch_size, batch_length =self.amount, self.batch, start_state_is_terminal.shape[1]
        _start_state = {k: tf.stack([v[:batch_size, :]]*amount, axis=0)
                                        for k, v in start_state.items()}
        _start_state_is_terminal = tf.stack([start_state_is_terminal[:batch_size,:]]*amount, axis=0)
        _start_state = {k:v.reshape((amount,batch_size*batch_length,)+ v.shape[3:]) for k, v in _start_state.items()}
        _start_state_is_terminal = _start_state_is_terminal.reshape((amount,batch_size*batch_length,)+ _start_state_is_terminal.shape[3:])
        mean = tf.constant((self.act_space.low + self.act_space.high)/2)
        std = tf.ones(mean.shape)
        mean = tf.tile(tf.expand_dims(tf.expand_dims(mean, axis=0), axis=0), [batch_size*batch_length,self.horizon+1,1])
        std = tf.tile(tf.expand_dims(tf.expand_dims(std, axis=0), axis=0), [batch_size*batch_length,self.horizon+1,1])
        for i in range(self.iteration):
            policy = MixPolicy(self.prior_mix, mean, std, actor_model, self.amount)
            seq = world_model.imagine(policy, _start_state, _start_state_is_terminal, self.horizon)
            reward = reward_fn(seq)
            if rewnorm is not None:
                seq['reward'], mets1 = rewnorm(reward)
            else:
                seq['reward'] = reward
            target, mets2 = target_func(seq)
            weight = tf.stop_gradient(seq['weight'])
            critic = target * weight[:-1]
            critic = critic[0,:].reshape((amount,batch_size*batch_length,))
            critic = tf.transpose(critic, [1,0])
            # sort_order = tf.argsort(critic, axis=1,direction='DESCENDING')
            # best_order = sort_order[:,:self.topk]
            best_order = tf.math.top_k(critic, k=self.topk, sorted=True,).indices
            actions = seq['action'].reshape((seq['action'].shape[0], amount, batch_size*batch_length,seq['action'].shape[2]))
            actions = tf.transpose(actions, [1,2,0,3])
            collects = []
            for k in range(best_order.shape[0]):
                collects.append(tf.gather(actions[:,k,:,:], best_order[k], axis=0))
            mean = tf.stack([tf.reduce_mean(v, axis=0) for v in collects],axis=0)
            std = tf.stack([tf.math.reduce_std(v, axis=0) for v in collects],axis=0)
        
        states = seq['feat'][:-1]
        states = states.reshape((states.shape[0],amount, batch_size*batch_length,states.shape[2]))
        states = tf.transpose(states, [2,1,0,3])
        states = states[:,0,:self.loss_horizon, :]
        actions = actor_model(tf.stop_gradient(states))    
        like = -tf.cast(actions.log_prob(mean[:,1:self.loss_horizon+1,:]), tf.float32).mean() 
        loss = like * self.loss_scale
        metrics = {'cem_loss': like}
        return loss, metrics