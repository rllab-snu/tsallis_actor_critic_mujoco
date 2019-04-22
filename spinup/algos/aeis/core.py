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

def tf_log_sum_exp(x):
    max_x = tf.reduce_max(x, axis=-1, keepdims=True)
    return tf.log(tf.reduce_sum(tf.exp(x - max_x), axis=-1)) + tf.reduce_max(x, axis=-1)

"""
Policies
"""

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp_q_gaussian_policy(x, a, q_prime, hidden_sizes, activation, output_activation, n_mixture=6):
    q = 2.0 - q_prime
    act_dim = a.shape.as_list()[-1]
    net = mlp(x, list(hidden_sizes), activation, activation)
    
    mu = tf.layers.dense(net, act_dim*n_mixture, activation=output_activation)
    mu = tf.reshape(mu, shape=[-1, act_dim, n_mixture], name="means")

    log_std = tf.layers.dense(net, act_dim*n_mixture, activation=output_activation)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    log_std = tf.reshape(log_std, shape=[-1, act_dim, n_mixture], name="log_stds")
    std = tf.exp(log_std)

    logit_weight = tf.layers.dense(net, n_mixture, activation=output_activation)
    weight = tf.nn.softmax(logit_weight)

    mask = tf.one_hot(tf.random.categorical(logit_weight, 1, dtype=tf.float32), n_mixture, on_value=1.0, off_value=0.0)
    mask = tf.reshape(mask, [-1, 1, n_mixture])

    pi = tf.reduce_sum(mask*(mu + tf.random_normal(tf.shape(mu)) * std), axis=-1)
    squashed_pi = tf.tanh(pi)

    exponents = -0.5 * (((pi[:, :, tf.newaxis]-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi)) \
                - tf.log(clip_but_pass_gradient(1 - squashed_pi**2, l=0, u=1)+1e-6) \
                - tf.log(weight)

    logp_pi = tf_log_sum_exp(exponents)
    q_logp_pi = tf.reduce_sum(tf_log_q(tf.exp(logp_pi), q), axis=1)

    mask = tf.one_hot(tf.argmax(logit_weight, -1), n_mixture, on_value=1.0, off_value=0.0)
    mask = tf.reshape(mask, [-1, 1, n_mixture])

    mu = tf.reduce_sum(mask*mu, axis=-1)
    squashed_mu = tf.tanh(mu)

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
