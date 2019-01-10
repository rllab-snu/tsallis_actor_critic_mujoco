import numpy as np
import tensorflow as tf
from spinup.algos.tac.tf_tsallis_statistics import *

EPS = 1e-8

def scale_holder():
    return tf.placeholder(dtype=tf.float32, shape=())

def entropic_index_holder():
    return tf.placeholder(dtype=tf.float32, shape=())

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)

"""
Policies
"""

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp_q_gaussian_policy(x, a, q_prime, hidden_sizes, activation, output_activation, pdf_type="gaussian", log_type="q-log"):
    q = 2.0 - q_prime
    act_dim = a.shape.as_list()[-1]
    net = mlp(x, list(hidden_sizes), activation, activation)
    
    mu = tf.layers.dense(net, act_dim, activation=output_activation)
    
    log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    std = tf.exp(log_std)
    
    if pdf_type=="gaussian" and log_type=="q-log":
        pi = mu + tf.random_normal(tf.shape(mu)) * std
        squashed_mu = tf.tanh(mu)
        squashed_pi = tf.tanh(pi)
        logp_pi = gaussian_likelihood(pi, mu, log_std) - tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - squashed_pi**2, l=1e-5, u=1)), axis=1)
        q_logp_pi = tf_log_q(tf.exp(logp_pi-tf.reduce_max(logp_pi)),q)
        
    if pdf_type=="gaussian" and log_type=="scaled-q-log":
        pi = mu + tf.random_normal(tf.shape(mu)) * std
        squashed_mu = tf.tanh(mu)
        squashed_pi = tf.tanh(pi)
        logp_pi = gaussian_likelihood(pi, mu, log_std) - tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - squashed_pi**2, l=1e-5, u=1)), axis=1)
        q_logp_pi = tf_log_q(tf.exp(logp_pi),q)*tf.log(tf.reduce_prod(1/2.0*tf.ones_like(mu),axis=1))/tf_log_q(tf.reduce_prod(1/2.0*tf.ones_like(mu),axis=1),q)
        
    elif pdf_type=="q-gaussian" and log_type=="q-log":
        pi = mu + tf_random_q_normal(tf.shape(mu),q) * std
        squashed_mu = tf.tanh(mu)
        squashed_pi = tf.tanh(pi)
        p_pi = tf_q_gaussian_distribution(pi, mu, log_std, q)/tf.reduce_prod(clip_but_pass_gradient(1 - squashed_pi**2, l=1e-5, u=1), axis=1)
        q_logp_pi = tf_log_q(p_pi,q)
    
    elif pdf_type=="q-gaussian" and log_type=="scaled-q-log":
        pi = mu + tf_random_q_normal(tf.shape(mu),q) * std
        squashed_mu = tf.tanh(mu)
        squashed_pi = tf.tanh(pi)
        p_pi = tf_q_gaussian_distribution(pi, mu, log_std, q)/tf.reduce_prod(clip_but_pass_gradient(1 - squashed_pi**2, l=1e-5, u=1), axis=1)
        q_logp_pi = tf_log_q(p_pi,q)*tf.log(tf.reduce_prod(1/2.0*tf.ones_like(mu),axis=1))/tf_log_q(tf.reduce_prod(1/2.0*tf.ones_like(mu),axis=1),q)
        
    elif pdf_type=="gaussian" and log_type=="log":
        pi = mu + tf.random_normal(tf.shape(mu)) * std
        squashed_mu = tf.tanh(mu)
        squashed_pi = tf.tanh(pi)
        q_logp_pi = gaussian_likelihood(pi, mu, log_std) - tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - squashed_pi**2, l=0, u=1) + 1e-6), axis=1)
    
    return squashed_mu, squashed_pi, q_logp_pi

"""
Actor-Critics
"""
def mlp_q_actor_critic(x, a, q, hidden_sizes=(400,300), activation=tf.nn.relu, 
                     output_activation=None, policy=mlp_q_gaussian_policy, pdf_type="gaussian", log_type="q-log", action_space=None):
    # policy
    with tf.variable_scope('pi'):
        mu, pi, q_logp_pi = policy(x, a, q, hidden_sizes, activation, output_activation, pdf_type=pdf_type, log_type=log_type)
        
    # make sure actions are in correct range
    action_scale = action_space.high[0]
    mu *= action_scale
    pi *= action_scale

    # vfs
    vf_mlp = lambda x : tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('q1'):
        q1 = vf_mlp(tf.concat([x,a], axis=-1))
    with tf.variable_scope('q1', reuse=True):
        q1_pi = vf_mlp(tf.concat([x,pi], axis=-1))
    with tf.variable_scope('q2'):
        q2 = vf_mlp(tf.concat([x,a], axis=-1))
    with tf.variable_scope('q2', reuse=True):
        q2_pi = vf_mlp(tf.concat([x,pi], axis=-1))
    with tf.variable_scope('v'):
        v = vf_mlp(x)
    return mu, pi, q_logp_pi, q1, q2, q1_pi, q2_pi, v
