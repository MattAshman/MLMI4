import os, ssl, sys

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and \
        getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

import pdb
import argparse
import numpy as np
import tensorflow as tf
import gpflow

from pathlib import Path
from gpflow.likelihoods import Gaussian
from gpflow.kernels import SquaredExponential, White
from gpflow.utilities import print_summary, triangular
from gpflow.base import Parameter
from scipy.cluster.vq import kmeans2
from scipy.stats import norm
from scipy.special import logsumexp

from datasets import Datasets
from dgp import DGP

def main(args):
    print('Getting dataset...')
    datasets = Datasets(data_path=args.data_path)
    data = datasets.all_datasets[args.dataset].get_data()
    X, Y, Xs, Ys, Y_std = [data[_] for _ in ['X', 'Y', 'Xs', 'Ys', 'Y_std']]
    Z = kmeans2(X, args.num_inducing, minit='points')[0]

    print('Setting up DGP model...')
    kernels = []
    for l in range(args.num_layers):
        kernels.append(SquaredExponential() + White(variance=2e-6))

    dgp_model = DGP(X.shape[1], kernels, Gaussian(variance=0.05), Z, 
            num_outputs=Y.shape[1], num_samples=args.num_samples)

    # initialise inner layers almost deterministically
    for layer in dgp_model.layers[:-1]:
        layer.q_sqrt = Parameter(layer.q_sqrt.value() * 1e-5, 
                transform = triangular())

    optimiser = tf.optimizers.Adam(args.learning_rate)

    def optimisation_step(model, X, Y):
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            obj = - model.elbo(X, Y, full_cov=False)
            grad = tape.gradient(obj, model.trainable_variables)
        optimiser.apply_gradients(zip(grad, model.trainable_variables))

    def monitored_training_loop(model, X, Y, logdir, epochs, 
            logging_epoch_freq):
        # TODO: use tensorboard to log trainables and performance
        tf_optimisation_step = tf.function(optimisation_step)

        for epoch in range(epochs):
            tf_optimisation_step(model, X, Y)

            epoch_id = epoch + 1
            if epoch_id % logging_epoch_freq == 0:
                tf.print(f'Epoch {epoch_id}: ELBO (train) {model.elbo(X, Y)}')

    print('Training DGP model...')
    monitored_training_loop(dgp_model, X, Y, logdir=args.log_dir, 
            epochs=args.epochs, logging_epoch_freq=args.logging_epoch_freq)

    test_ll = dgp_model.log_likelihood(Xs, Ys)
    print('Average test log likelihood: {}'.format(test_ll / Xs.shape[0]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data/', 
        help='Path to datafile.')
    parser.add_argument('--dataset', help='Name of dataset to run.')
    parser.add_argument('--num_inducing', type=int, default=50, 
        help='Number of inducing input locations.')
    parser.add_argument('--num_layers', type=int, default=3,
        help='Number of DGP layers.')
    parser.add_argument('--num_samples', type=int, default=1,
        help='Number of samples to propagate.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
        help='Learning rate for optimiser.')
    parser.add_argument('--epochs', type=int, default=2000, 
        help='Number of training epochs.')
    parser.add_argument('--log_dir', default='./log/', 
        help='Directory log files are written to.')
    parser.add_argument('--logging_epoch_freq', type=int, default=100,
        help='Number of epochs between training logs.')

    args = parser.parse_args()
    main(args)
