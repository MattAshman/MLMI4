import pdb
import time
import os, ssl, sys
sys.path.append('../code/')

import numpy as np
import tensorflow as tf
import gpflow
import argparse

from gpflow.likelihoods import Gaussian
from gpflow.kernels import SquaredExponential, White
from gpflow.utilities import print_summary
from gpflow.base import Parameter
from gpflow.config import default_float
from scipy.cluster.vq import kmeans2
from scipy.stats import norm
from scipy.special import logsumexp

from datasets import Datasets

def main(args):
    datasets = Datasets(data_path=args.data_path)

    # Prepare output files
    outname1 = '../svgp_tmp/svgp_' + args.dataset + '_' + str(args.num_inducing) + '.rmse'
    if not os.path.exists(os.path.dirname(outname1)):
        os.makedirs(os.path.dirname(outname1))
    outfile1 = open(outname1, 'w')
    outname2 = '../svgp_tmp/svgp_' + args.dataset + '_' + str(args.num_inducing) + '.nll'
    outfile2 = open(outname2, 'w')
    outname3 = '../svgp_tmp/svgp_' + args.dataset + '_' + str(args.num_inducing) + '.time'
    outfile3 = open(outname3, 'w')

    def optimisation_step(model, X, Y, optimizer):
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            obj = - model.elbo((X, Y))
            grads = tape.gradient(obj, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    def monitored_training_loop(model, train_dataset, logdir, iterations,
                                logging_iter_freq, optimizer):
        batches = iter(train_dataset)
        tf_optimisation_step = tf.function(optimisation_step)

        for i in range(iterations):
            X, Y = next(batches)
            tf_optimisation_step(model, X, Y, optimizer)

            iter_id = i + 1
            if iter_id % logging_iter_freq == 0:
                print('Epoch {}: ELBO (batch) {}'.format(iter_id,
                                                         model.elbo((X, Y))))

    running_err = 0
    running_loss = 0
    running_time = 0
    test_errs = np.zeros(args.splits)
    test_nlls = np.zeros(args.splits)
    test_times = np.zeros(args.splits)
    for i in range(args.splits):
        print('Split: {}'.format(i))
        print('Getting dataset...')
        data = datasets.all_datasets[args.dataset].get_data(i, normalize=args.normalize_data)
        X, Y, Xs, Ys, Y_std = [data[_] for _ in ['X', 'Y', 'Xs', 'Ys', 'Y_std']]
        Z = kmeans2(X, args.num_inducing, minit='points')[0]

        # set up batches
        batch_size = args.batch_size if args.batch_size < X.shape[0]\
            else X.shape[0]
        train_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).repeat()\
                .prefetch(X.shape[0]//2)\
                .shuffle(buffer_size=(X.shape[0]//2))\
                .batch(batch_size)

        print('Setting up SVGP model...')
        kernel = SquaredExponential()
        likelihood = Gaussian(variance=0.05)
        model = gpflow.models.SVGP(kernel=kernel, likelihood=likelihood,
                                   inducing_variable=Z)

        print('Training SVGP model...')
        optimizer = tf.optimizers.Adam(args.learning_rate)
        t0 = time.time()
        monitored_training_loop(model, train_dataset, logdir=args.log_dir,
                                iterations=args.iterations,
                                logging_iter_freq=args.logging_iter_freq,
                                optimizer=optimizer)
        t1 = time.time()
        test_times[i] = t1 - t0
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
                m, v = model.predict_y(
                        Xs[mb*test_batch_size:(mb+1)*test_batch_size, :])
                means.append(m)
                vars.append(v)
        else:
            m, v = dgp_model.predict_y(Xs)
            means.append(m)
            vars.append(v)

        mean_ND = np.concatenate(means, 1)
        var_ND = np.concatenate(vars, 1)

        test_err = np.mean(Y_std * np.mean((Ys - mean_ND) ** 2.0) ** 0.5)
        test_errs[i] = test_err
        print('Average RMSE: {}'.format(test_err))
        outfile1.write('Split {}: {}\n'.format(i+1, test_err))
        outfile1.flush()
        os.fsync(outfile1.fileno())
        running_err += test_err

        test_nll = np.mean(norm.logpdf(Ys * Y_std, mean_ND * Y_std,
            var_ND ** 0.5 * Y_std))
        test_nlls[i] = test_nll
        print('Average test log likelihood: {}'.format(test_nll))
        outfile2.write('Split {}: {}\n'.format(i+1, test_nll))
        outfile2.flush()
        os.fsync(outfile2.fileno())
        running_loss += test_nll

    outfile1.write('Average: {}\n'.format(running_err / args.splits))
    outfile1.write('Standard deviation: {}\n'.format(np.std(test_errs)))
    outfile2.write('Average: {}\n'.format(running_loss / args.splits))
    outfile2.write('Standard deviation: {}\n'.format(np.std(test_nlls)))
    outfile3.write('Average: {}\n'.format(running_time / args.splits))
    outfile3.write('Standard deviation: {}\n'.format(np.std(test_times)))
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
    parser.add_argument('--learning_rate', type=float, default=0.01,
        help='Learning rate for optimiser.')
    parser.add_argument('--iterations', type=int, default=20000,
        help='Number of training iterations.')
    parser.add_argument('--log_dir', default='./log/',
        help='Directory log files are written to.')
    parser.add_argument('--logging_iter_freq', type=int, default=500,
        help='Number of iterations between training logs.')
    parser.add_argument('--batch_size', type=int, default=10000,
        help='Minibatch size.')
    parser.add_argument('--test_batch_size', type=int, default=100,
        help='Batch size to apply to test data.')
    parser.add_argument('--normalize_data', type=bool, default=True,
        help='Whether or not to normalize the data.')

    args = parser.parse_args()
    main(args)
