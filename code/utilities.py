import pdb
import numpy as np
import tensorflow as tf
import gpflow
from gpflow.config import default_float, default_jitter

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-6)

def reparameterise(mean, var, z, full_cov=False):
    """Implements the reparameterisation trick for the Gaussian, either full
    rank or diagonal.

    If z is a sample from N(0,I), the output is a sample from N(mean,var).

    :mean: A tensor, the mean of shape [S,N,D].
    :var: A tensor, the coariance of shape [S,N,D] or [S,N,N,D].
    :z: A tensor, samples from a unit Gaussian of shape [S,N,D].
    :full_cov: A boolean, indicates the shape of var."""
    if var is None:
        return mean

    if full_cov is False:
        return mean + z * (var + default_jitter()) ** 0.5

    else:
        S, N, D = tf.shape(mean)[0], tf.shape(mean)[1], tf.shape(mean)[2]
        mean = tf.transpose(mean, (0, 2, 1)) # [S,N,D]
        var = tf.transpose(var, (0, 3, 1, 2)) # [S, D, N, N]
        I = default_jitter() * tf.eye(N, dtype=default_float())\
                [None, None, :, :] # [1,1,N,N]
        chol = tf.linalg.cholesky(var + I)
        z_SDN1 = tf.transpose(z, (0, 2, 1))[:, :, :, None]
        f = mean + tf.matmul(chol, z_SDN1)[:, :, :, 0]
        return tf.transpose(f, (0, 2, 1)) # [S,N,D]
