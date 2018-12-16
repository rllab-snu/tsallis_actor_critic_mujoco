import numpy as np
import tensorflow as tf
from spinup.algos.tac.tf_tsallis_statistics import *

EPS = 1e-8

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
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 - 2*log_std - np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)


"""
Policies
"""

LOG_STD_MAX = 6
LOG_STD_MIN = -4

def mlp_q_gaussian_policy(x, a, q, hidden_sizes, activation, output_activation):
    act_dim = a.shape.as_list()[-1]
    net = mlp(x, list(hidden_sizes), activation, activation)
    mu = tf.layers.dense(net, act_dim, activation=output_activation)

    """
    Because algorithm maximizes trade-off of reward and entropy,
    entropy must be unique to state---and therefore log_stds need
    to be a neural network output instead of a shared-across-states
    learnable parameter vector. But for deep Relu and other nets,
    simply sticking an activationless dense layer at the end would
    be quite bad---at the beginning of training, a randomly initialized
    net could produce extremely large values for the log_stds, which
    would result in some actions being either entirely deterministic
    or too random to come back to earth. Either of these introduces
    numerical instability which could break the algorithm. To 
    protect against that, we'll constrain the output range of the 
    log_stds, to lie within [LOG_STD_MIN, LOG_STD_MAX]. This is 
    slightly different from the trick used by the original authors of
    SAC---they used tf.clip_by_value instead of squashing and rescaling.
    I prefer this approach because it allows gradient propagation
    through log_std where clipping wouldn't, but I don't know if
    it makes much of a difference.
    """
    log_invbeta = tf.layers.dense(net, act_dim, activation=tf.tanh)
    log_invbeta = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_invbeta + 1)

    invbeta = tf.exp(log_invbeta)
    pi = mu + tf_random_q_normal(tf.shape(mu),q) * invbeta

    squashed_mu = tf.tanh(mu)
    squashed_pi = tf.tanh(pi)

    q_logp_pi = tf_log_q(tf_q_gaussian_distribution(pi, mu, log_invbeta, q),q=q) - tf.reduce_sum(tf_log_q(clip_but_pass_gradient(1 - squashed_pi**2, l=0, u=1) + 1e-8, q=q), axis=1)
#    q_logp_pi = tf_log_q(tf_q_gaussian_distribution(pi, mu, log_invbeta, q)/tf.reduce_prod(clip_but_pass_gradient(1 - squashed_pi**2, l=0, u=1) + 1e-4, axis=1),q=q)
    return squashed_mu, squashed_pi, q_logp_pi

#    log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
#    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

#    std = tf.exp(log_std)
#    pi = mu + tf.random_normal(tf.shape(mu)) * std
#    logp_pi = gaussian_likelihood(pi, mu, log_std)
    
#    squashed_mu = tf.tanh(mu)
#    squashed_pi = tf.tanh(pi)
    
#    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - squashed_pi**2, l=0, u=1) + 1e-6), axis=1)
    
#    return squashed_mu, squashed_pi, logp_pi

"""
Actor-Critics
"""
def mlp_q_actor_critic(x, a, q, hidden_sizes=(400,300), activation=tf.nn.relu, 
                     output_activation=None, policy=mlp_q_gaussian_policy, action_space=None):
    # policy
    with tf.variable_scope('pi'):
        mu, pi, q_logp_pi = policy(x, a, q, hidden_sizes, activation, output_activation)
        
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