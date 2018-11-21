import tensorflow as tf
import numpy as np

def np_exp_q(x,q=1):
    if q==1:
        return np.exp(x)
    else:
        exp_q_x = np.zeros_like(x)
        exp_q_x[1+(1-q)*x > 0] = np.power(1+(1-q)*x[1+(1-q)*x > 0],1/(1-q))
        return exp_q_x

def np_log_q(x,q=1):
    if q==1:
        log_q_x = np.full_like(x,-np.inf)
        log_q_x[x>0] = np.log(x[x>0])
        return log_q_x
    else:
        log_q_x = np.full_like(x,-np.inf)
        log_q_x[x>0] = (np.power(x[x>0],1-q)-1)/(1-q)
        return log_q_x

def tf_exp_q(x,q):
    exp_q_x = tf.cond(tf.equal(q,1),true_fn=lambda: tf.exp(x),false_fn=lambda: tf.pow(tf.maximum(1+(q-1)*x,0),1/(q-1)))
    return exp_q_x
    
def tf_log_q(x,q):
    safe_x = tf.maximum(x,1e-5)
    log_q_x = tf.cond(tf.equal(q,1),true_fn=lambda: tf.log(safe_x),false_fn=lambda: (tf.pow(safe_x,q-1)-1)/(q-1))
    return log_q_x

def tf_tsallis_entropy(p,q):
    return tf.reduce_sum(-tf_log_q(p,q)*p,axis=1,keepdims=True)

def tf_tsallis_divergence_with_logits(p1,p2_q_logits,q):
    return tf.reduce_sum((tf_log_q(p1,q)-p2_q_logits)*p1,axis=1,keepdims=True)
    
def tf_tsallis_divergence(p1,p2,q):
    return tf.reduce_sum((tf_log_q(p1,q)-tf_log_q(p2,q))*p1,axis=1,keepdims=True)
    
def tf_tsallis_distance(p1,p2,q):
    return tf_tsallis_entropy((p1+p2)/2,q) - (tf_tsallis_entropy(p1,q)+tf_tsallis_entropy(p2,q))/2

def tf_random_q_normal(shape,q):
    q_prime = (3-q) / (1+q)
    U1 = tf.random_uniform(shape)
    U2 = tf.random_uniform(shape)
    return tf.sqrt(-2.*tf_log_q(U1,q=q))*tf.cos(tf.constant(2*np.pi)*U2)

def tf_q_gaussian_distribution(x, mu, log_invbeta, q):
    gamma1 = tf.exp(tf.lgamma(1/(q-1)))
    gamma2 = tf.exp(tf.lgamma((1+q)/2/(q-1)))

    C_q = tf.cond(tf.equal(q,1),true_fn=lambda: tf.constant(1./np.sqrt(2.),dtype=tf.float32),false_fn=lambda: tf.sqrt(tf.constant(2.,dtype=tf.float32)) * gamma1 /gamma2 /tf.sqrt(q-1)/(1+q))
    
    tf_pdf = tf_exp_q(-0.5*tf.square((x-mu) /(tf.exp(log_invbeta)+1e-8)),q=q) /C_q / tf.sqrt(tf.constant(2.*np.pi,dtype=tf.float32))
    return tf_pdf
    
    
    
    
    
    
    