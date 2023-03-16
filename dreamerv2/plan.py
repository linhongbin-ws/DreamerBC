import tensorflow as tf
from tensorflow.keras import mixed_precision as prec
import common
import numpy as np
from tensorflow_probability import distributions as tfd

class MixPolicy(object):
    def __init__(self, mix, mean, std, prior_policy, amount):
        self.mix = mix
        self.mean = tf.repeat(mean, repeats=amount, axis=0)
        self.std = tf.repeat(std, repeats=amount, axis=0)
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
                amount=100,
                topk=10,
                iteration=10,
                prior_mix=0.5,
                batch=8,
                ):
        self.act_space = act_space
        self.horizon = horizon
        self.amount = amount
        self.topk=topk
        self.iteration = iteration
        self.prior_mix = prior_mix
        self.batch = batch

    def plan(self, 
            world_model, 
            actor_model, 
            start_state, 
            start_state_is_terminal, 
            target_func,
            reward_fn,
            rewnorm,
                ):
        
        _start_state = {k: tf.reshape(tf.stack([v[:self.batch, :]]*self.amount, axis=1), self.batch*self.amount+v.shape[1:]) for k, v in start_state.items()}
        _start_state_is_terminal = tf.reshape(tf.stack([start_state_is_terminal[:self.batch, :]]*self.amount, axis=1), self.batch*self.amount+start_state_is_terminal.shape[1:]) 
        

        mean = tf.constant((self.act_space.low + self.act_space.high)/2)
        std = tf.ones(mean.shape)
        mean = tf.tile(tf.expand_dims(tf.expand_dims(mean, axis=0), axis=0), [self.batch,self.horizon+1,1])
        std = tf.tile(tf.expand_dims(tf.expand_dims(std, axis=0), axis=0), [self.batch,self.horizon+1,1])
        for i in range(self.iteration):
            policy = MixPolicy(self.prior_mix, mean, std, actor_model, self.amount)
            seq = world_model.imagine(policy, start_state, start_state_is_terminal, self.horizon)
            reward = reward_fn(seq)
            if rewnorm is not None:
                seq['reward'], mets1 = rewnorm(reward)
            else:
                seq['reward'] = reward
            target, mets2 = target_func(seq)
            weight = tf.stop_gradient(seq['weight'])
            critic = target * weight[:-1]
            critic = critic[0,:].reshape((self.batch,self.amount))
            sort_order = tf.argsort(critic, axis=1)
            best_order = sort_order[:,:self.topk]
            actions = seq['action'].reshape((seq['action'].shape[0],self.batch, self.amount,seq['action'].shape[2]))
            actions = tf.transpose(actions, [1,2,0,3])
            collects = []
            for k in range(best_order.shape[0]):
                collects.append(tf.gather(actions[k,:], best_order[k], axis=0))
             
        return action, loss