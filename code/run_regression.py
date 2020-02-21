import os, ssl, sys

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and \
        getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

import pdb
import argparse
import time
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
    datasets = Datasets(data_path=args.data_path)

    # Prepare output files
    outname1 = '../tmp/' + args.dataset + '_' + str(args.num_layers) + '_'\
            + str(args.num_inducing) + '.rmse'
    if not os.path.exists(os.path.dirname(outname1)):
        os.makedirs(os.path.dirname(outname1))
    outfile1 = open(outname1, 'w')
    outname2 = '../tmp/' + args.dataset + '_' + str(args.num_layers) + '_'\
            + str(args.num_inducing) + '.nll'
    outfile2 = open(outname2, 'w')
    outname3 = '../tmp/' + args.dataset + '_' + str(args.num_layers) + '_'\
            + str(args.num_inducing) + '.time'
    outfile3 = open(outname3, 'w')

    running_err = 0
    running_loss = 0
    running_time = 0
    for i in range(args.splits):
        print('Split: {}'.format(i))
        print('Getting dataset...')
        data = datasets.all_datasets[args.dataset].get_data(i, normalize=args.normalize_data)
        X, Y, Xs, Ys, Y_std = [data[_] for _ in ['X', 'Y', 'Xs', 'Ys', 'Y_std']]
        Z = kmeans2(X, args.num_inducing, minit='points')[0]

        # set up batches
        batch_size = args.M if args.M < X.shape[0] else X.shape[0]
        train_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).repeat()\
                .prefetch(X.shape[0]//2)\
                .shuffle(buffer_size=(X.shape[0]//2))\
                .batch(batch_size)

        print('Setting up DGP model...')
        kernels = []
        dims = []
        hidden_dim = X.shape[1] if X.shape[1] < args.max_dim else args.max_dim
        for l in range(args.num_layers):
            kernels.append(SquaredExponential() + White(variance=1e-5))
            if l == 0:
                dims.append(X.shape[1])
            else:
                dims.append(hidden_dim)
        dims.append(Y.shape[1])

        dgp_model = DGP(X, Y, Z, dims, kernels, Gaussian(variance=0.05),
                num_samples=args.num_samples, num_data=X.shape[0])

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

        def monitored_training_loop(model, train_dataset, logdir, iterations, 
                logging_iter_freq):
            # TODO: use tensorboard to log trainables and performance
            tf_optimisation_step = tf.function(optimisation_step)
            batches = iter(train_dataset)

            for i in range(iterations):
                X, Y = next(batches)
                tf_optimisation_step(model, X, Y)

                iter_id = i + 1
                if iter_id % logging_iter_freq == 0:
                    print('Epoch {}: ELBO (batch) {}'.format(iter_id,
                        model.elbo(X, Y)))

        print('Training DGP model...')
        t0 = time.time()
        monitored_training_loop(dgp_model, train_dataset, logdir=args.log_dir, 
                iterations=args.iterations, 
                logging_iter_freq=args.logging_iter_freq)
        t1 = time.time()
        print('Time taken to train: {}'.format(t1 - t0))
        outfile3.write('Split {}: {}\n'.format(i+1, t1-t0))
        outfile3.flush()
        os.fsync(outfile3.fileno())
        running_time += t1 - t0

        # Minibatch test predictions
        means, vars = [], []
        test_batch_size = args.test_batch_size
        if len(Xs) > test_batch_size:
            for mb in range(-(-len(Xs) // test_batch_size)):
                m, v = dgp_model.predict_y(
                        Xs[mb*test_batch_size:(mb+1)*test_batch_size, :],
                        num_samples=args.test_samples)
                means.append(m)
                vars.append(v)
        else:
            m, v = dgp_model.predict_y(Xs, num_samples=args.test_samples)
            means.append(m)
            vars.append(v)

        mean_SND = np.concatenate(means, 1)
        var_SND = np.concatenate(vars, 1)
        mean_ND = np.mean(mean_SND, 0)
        
        test_err = np.mean(Y_std * np.mean((Ys - mean_ND) ** 2.0) ** 0.5)
        print('Average RMSE: {}'.format(test_err))
        outfile1.write('Split {}: {}\n'.format(i+1, test_err))
        outfile1.flush()
        os.fsync(outfile1.fileno())
        running_err += test_err

        test_nll = np.mean(logsumexp(norm.logpdf(Ys * Y_std, mean_SND * Y_std, 
                var_SND ** 0.5 * Y_std), 0, b=1 / float(args.test_samples)))
        print('Average test log likelihood: {}'.format(test_nll))
        outfile2.write('Split {}: {}\n'.format(i+1, test_nll))
        outfile2.flush()
        os.fsync(outfile2.fileno())
        running_loss += test_nll
    
    outfile1.write('Average: {}\n'.format(running_err / args.splits))
    outfile2.write('Average: {}\n'.format(running_loss / args.splits))
    outfile3.write('Average: {}\n'.format(running_time / args.splits))
    outfile1.close()
    outfile2.close()
    outfile3.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', default=20, type=int, 
        help='Number of cross-validation splits.')
    parser.add_argument('--data_path', default='../data/', 
        help='Path to datafile.')
    parser.add_argument('--dataset', help='Name of dataset to run.')
    parser.add_argument('--num_inducing', type=int, default=100, 
        help='Number of inducing input locations.')
    parser.add_argument('--num_layers', type=int, default=2,
        help='Number of DGP layers.')
    parser.add_argument('--num_samples', type=int, default=1,
        help='Number of samples to propagate.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
        help='Learning rate for optimiser.')
    parser.add_argument('--iterations', type=int, default=10000, 
        help='Number of training iterations.')
    parser.add_argument('--log_dir', default='./log/', 
        help='Directory log files are written to.')
    parser.add_argument('--logging_iter_freq', type=int, default=500,
        help='Number of iterations between training logs.')
    parser.add_argument('--M', type=int, default=10000, 
        help='Minibatch size.')
    parser.add_argument('--test_samples', type=int, default=100, 
        help='Number of test samples to use.')
    parser.add_argument('--test_batch_size', type=int, default=100,
        help='Batch size to apply to test data.')
    parser.add_argument('--max_dim', type=int, default=30,
        help='Maximum dimension of latent variables.')
    parser.add_argument('--normalize_data', type=bool, default=True,
        help='Whether or not to normalize the data.')

    args = parser.parse_args()
    main(args)
