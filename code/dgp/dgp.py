import pdb
import numpy as np
import tensorflow as tf
import gpflow
from gpflow.models import BayesianModel
from gpflow.mean_functions import Linear, Identity, Zero
from gpflow.config import default_float, default_jitter
from gpflow.utilities import set_trainable
from layers import SVGPLayer, SVGPIndependentLayer

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-6)

class DGPBase(BayesianModel):
    """Base class for deep gaussian processes."""

    def __init__(self, likelihood, layers, num_samples=10, num_data=None, 
            **kwargs):
        super().__init__(**kwargs)

        self.likelihood = likelihood
        self.layers = layers
        self.num_samples = num_samples # Is this needed here?
        self.num_data = num_data

    def propagate(self, X, full_cov=False, S=1, zs=None):
        """Propagate input X through layers of the DGP S times. 

        :X: A tensor, the input to the DGP.
        :full_cov: A bool, indicates whether or not to use the full
        covariance matrix.
        :S: An int, the number of samples to draw.
        :zs: A tensor, samples from N(0,1) to use in the reparameterisation
        trick."""
        sX = tf.tile(tf.expand_dims(X, 0), [S, 1, 1]) # [S,N,D]
        Fs, Fmeans, Fvars = [], [], []
        F = sX
        zs = zs or [None, ] * len(self.layers) # [None, None, ..., None]
        for layer, z in zip(self.layers, zs):
            F, Fmean, Fvar = layer.sample_from_conditional(F, z=z,
                    full_cov=full_cov)

            Fs.append(F)
            Fmeans.append(Fmean)
            Fvars.append(Fvar)

        return Fs, Fmeans, Fvars

    def _predict(self, X, full_cov=False, S=1):
        Fs, Fmeans, Fvars = self.propagate(X, full_cov=full_cov, S=S)
        return Fmeans[-1], Fvars[-1]

    def E_log_p_Y(self, X, Y, full_cov=False):
        """Computes Monte Carlo estimate of the expected log density of the
        data, given a Gaussian distribution for the function values.

        if 

            q(f) = N(Fmu, Fvar)
            
        this method approximates

            \int (\log p(y|f)) q(f) df"""
        Fmean, Fvar = self._predict(X, full_cov=full_cov, S=self.num_samples)
        var_exp = self.likelihood.variational_expectations(Fmean, Fvar, Y)
        return tf.reduce_mean(var_exp, 0)

    def prior_kl(self):
        return tf.reduce_sum([layer.KL() for layer in self.layers])

    def log_likelihood(self, X, Y, full_cov=False, num_batches=None):
        """Gives a variational bound on the model likelihood."""
        # No batches for now
        L = tf.reduce_sum(self.E_log_p_Y(X, Y, full_cov))
        KL = self.prior_kl()
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, KL.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], KL.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, KL.dtype)

        return L * scale - KL

    def elbo(self, X, Y, full_cov=False):
        """ This returns the evidence lower bound (ELBO) of the log 
        marginal likelihood. """
        return self.log_marginal_likelihood(X=X, Y=Y, full_cov=full_cov) 

    def predict_f(self, Xnew, num_samples, full_cov=False):
        """Returns mean and variance of the final layer."""
        return self._predict(Xnew, full_cov=full_cov, S=num_samples)

    def predict_y(self, Xnew, num_samples, full_cov=False):
        Fmean, Fvar = self._predict(Xnew, full_cov=full_cov, S=num_samples)
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)

    def predict_all_layers(self, Xnew, num_samples, full_cov=False):
        """Returns mean and variance of all layers."""
        return self.propagate(Xnew, full_cov=full_cov, S=num_samples)

    def predict_density(self, Xnew, Ynew, num_samples, full_cov=False):
        Fmean, Fvar = self._predict(Xnew, full_cov=full_cov, S=num_samples)
        l = self.likelihood.predict_density(Fmean, Fvar, Ynew)
        log_num_samples = tf.log(tf.cast(num_samples, l.dtype))
        return tf.reduce_logsumexp(l - log_num_samples, axis=0)

class DGP(DGPBase):
    """The Doubly-Stochastic Deep GP, with linear/identity mean functions at
    each layer."""
    
    def __init__(self, X, Y, Z, dims, kernels, likelihood, 
            mean_function=Zero(), white=False, **kwargs):

        layers = self._init_layers(X, Y, Z, dims, kernels, 
                mean_function=mean_function, white=white)

        super().__init__(likelihood, layers, **kwargs)
        
    def _init_layers(self, X, Y, Z, dims, kernels, mean_function=Zero(),
            Layer=SVGPLayer, white=False):
        """Initialise DGP layers to have the same number of outputs as inputs,
        apart from the final layer."""
        layers = []

        X_running, Z_running = X.copy(), Z.copy()
        for i in range(len(kernels)-1):
            dim_in, dim_out, kern = dims[i], dims[i+1], kernels[i]
            if dim_in == dim_out:
                mf = Identity()

            else:
                if dim_in > dim_out:
                    _, _, V = np.linalg.svd(X_running, full_matrices=False)
                    W = V[:dim_out, :].T

                else:
                    W = np.concatenate([np.eye(dim_in), np.zeros((dim_in, 
                        dim_out - dim_in))], 1)

                mf = Linear(W)
                set_trainable(mf.A, False)
                set_trainable(mf.b, False)

            layers.append(Layer(kern, Z_running, dim_out, mf, white=white))

            if dim_in != dim_out:
                Z_running = Z_running.dot(W)
                X_running = X_running.dot(W)

        layers.append(Layer(kernels[-1], Z_running, dims[-1], mean_function, 
            white=white))
        return layers

class DGPIndependent(DGPBase):
    """The Doubly-Stochastic Deep GP, with linear/identity mean functions at
    each layer."""
    
    def __init__(self, X, Y, Z, dims, kernels, likelihood, 
            mean_function=Zero(), white=False, **kwargs):

        layers = self._init_layers(X, Y, Z, dims, kernels, 
                mean_function=mean_function, white=white)

        super().__init__(likelihood, layers, **kwargs)
        
    def _init_layers(self, X, Y, Z, dims, kernels, mean_function=Zero(),
            Layer=SVGPIndependentLayer, white=False):
        """Initialise DGP layers to have the same number of outputs as inputs,
        apart from the final layer."""
        layers = []

        X_running, Z_running = X.copy(), Z.copy()
        for i in range(len(kernels)-1):
            dim_in, dim_out, kern = dims[i], dims[i+1], kernels[i]
            if dim_in == dim_out:
                mf = Identity()

            else:
                if dim_in > dim_out:
                    _, _, V = np.linalg.svd(X_running, full_matrices=False)
                    W = V[:dim_out, :].T

                else:
                    W = np.concatenate([np.eye(dim_in), np.zeros((dim_in, 
                        dim_out - dim_in))], 1)

                mf = Linear(W)
                set_trainable(mf.A, False)
                set_trainable(mf.b, False)

            layers.append(Layer(kern, Z_running, dim_out, mf, white=white))

            if dim_in != dim_out:
                Z_running = Z_running.dot(W)
                X_running = X_running.dot(W)

        layers.append(Layer(kernels[-1], Z_running, dims[-1], mean_function, 
            white=white))
        return layers
