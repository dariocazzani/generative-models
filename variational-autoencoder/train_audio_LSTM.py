#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
Authors:    Dario Cazzani
"""
import sys
sys.path.append('../')
from config import set_config
from helpers.misc import check_tf_version, extend_options
from helpers.graph import get_variables, linear, AdamOptimizer
from helpers.signal_processing import floats_to_wav

import subprocess
import tensorflow as tf
import numpy as np
import os
import glob

from helpers.data_dispatcher import CMajorScaleDistribution, NSynthGenerator, SinusoidDistribution

# Parameters
input_dim = 1600
num_frames = 10
frame_length = 480
frame_step = 160
fft_length= 512
hidden_layer = 128

# Q(z|X)
def encoder(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.variable_scope('Encoder'):
        e_cell1 = tf.nn.rnn_cell.LSTMCell(hidden_layer)
        _, states = tf.contrib.rnn.static_rnn(e_cell1, x, dtype=tf.float32)
        state_c, state_h = states
        hidden_state = tf.concat([state_c, state_h], 1)
        z_mu = linear(hidden_state, options.z_dim, 'z_mu')
        z_logvar = linear(hidden_state, options.z_dim, 'z_logvar')
        return z_mu, z_logvar

def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps

# P(X|z)
def decoder(z, inputs=None, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.variable_scope('Decoder'):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(input_dim, state_is_tuple=False)
        expanded_z = tf.nn.relu(linear(z, input_dim*2, 'linear_expand_z'))
        first_input = tf.add(tf.zeros_like(expanded_z[:, :input_dim]), 1.)
        output, state = lstm_cell(first_input, expanded_z)
        outputs = []
        outputs.append(output)
        for i in range(num_frames-1):
            # Training
            if inputs:
                output, state = lstm_cell(inputs[i], state)
            else:
                output, state = lstm_cell(output, state)
            outputs.append(output)

        logits = tf.stack(outputs, axis=1, name='logits')
        logits_reshaped = tf.reshape(logits, [-1, num_frames * input_dim])
        out = tf.nn.tanh(logits_reshaped)
        return out, logits_reshaped


def train(options):
    audiofiles = glob.glob(options.DATA_PATH + '/nsynth-test/audio/*wav')
    data = CMajorScaleDistribution(options.batch_size)
    # data = NSynthGenerator(audiofiles, options.batch_size)

    # Placeholders for input data and the targets
    with tf.name_scope('Input'):
        X = tf.placeholder(dtype=tf.float32, shape=[None, input_dim * num_frames], name='Input')
        X_reshaped = tf.reshape(X, [-1, num_frames, input_dim])
        # Unstack to get a list of 'num_frames' tensors of shape (batch_size, input_dim)
        X_list = tf.unstack(X_reshaped, num_frames, 1)

    with tf.name_scope('Latent_variable'):
        z = tf.placeholder(dtype=tf.float32, shape=[None, options.z_dim], name='Latent_variable')

    with tf.name_scope('Autoencoder'):
        with tf.variable_scope(tf.get_variable_scope()):
            z_mu, z_logvar = encoder(X_list)
            z_sample = sample_z(z_mu, z_logvar)
            decoder_output, logits = decoder(z_sample, inputs=None)

    with tf.variable_scope(tf.get_variable_scope()):
        X_samples, _ = decoder(z, reuse=True)

    # Loss
    with tf.name_scope('Loss'):
        # E[log P(X|z)]
        normalized_X = tf.div(tf.add(X, 1.), 2.)
        reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=normalized_X), 1)
        # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
        kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
        # VAE loss
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

    # Optimizer
    train_op, grads_and_vars = AdamOptimizer(vae_loss, options.learning_rate, options.beta1)

    # Visualization
    tf.summary.scalar(name='Loss', tensor=vae_loss)
    tf.summary.histogram(name='Encoder z_mu', values=z_mu)
    tf.summary.histogram(name='Encoder z_logvar', values=z_logvar)
    tf.summary.histogram(name='Sampled variable', values=z_sample)

    for grad, var in grads_and_vars:
        tf.summary.histogram(var.name + '/gradient', grad)
        tf.summary.histogram(var.name + '/value', var)

    tf.summary.audio(name='Input Sounds', tensor=X, sample_rate = 16000, max_outputs=3)
    tf.summary.audio(name='Generated Sounds', tensor=decoder_output, sample_rate = 16000, max_outputs=3)
    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    step = 0
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        if not options.run_inference:
            try:
                writer = tf.summary.FileWriter(logdir=options.tensorboard_path, graph=sess.graph)
                for i in range(options.epochs):

                    batch_x = data.__next__()
                    sess.run(train_op, feed_dict={X: batch_x})
                    if i % 50 == 0:
                        batch_loss, summary = sess.run([vae_loss, summary_op], feed_dict={X: batch_x})
                        writer.add_summary(summary, global_step=step)
                        print("Epoch: {} - Loss: {}\n".format(i, batch_loss))
                        with open(options.logs_path + '/log.txt', 'a') as log:
                            log.write("Epoch: {} - Loss: {}\n".format(i, batch_loss))

                        saver.save(sess, save_path=options.checkpoints_path, global_step=step)

                    step += 1

                print("Model Trained!")
                print("Tensorboard Path: {}".format(options.tensorboard_path))
                print("Log Path: {}".format(options.logs_path + '/log.txt'))
                print("Saved Model Path: {}".format(options.checkpoints_path))
            except KeyboardInterrupt:
                print('Stopping training...')
                print("Saved Model Path: {}".format(options.checkpoints_path))
                saver.save(sess, save_path=options.checkpoints_path, global_step=step)
        else:
            if not os.path.exists('out/'):
                os.makedirs('out/')

            print('Restoring latest saved TensorFlow model...')
            dir_path = os.path.dirname(os.path.realpath(__file__))
            cur_dir = dir_path.split('/')[-1]
            script_name = os.path.basename(__file__).split('.')[0]
            experiments = glob.glob(os.path.join(options.MAIN_PATH, cur_dir) + '/{}*'.format(script_name))
            experiments.sort(key=lambda x: os.path.getmtime(x))
            if len(experiments) > 0:
                print('Restoring: {}'.format(experiments[-1]))
                saver.restore(sess, tf.train.latest_checkpoint(os.path.join(experiments[-1], 'checkpoints')))

                samples = []
                n = 10
                z_ = []
                grid_x = np.linspace(-1, 1, n)
                grid_y = np.linspace(-1, 1, n)
                for i, yi in enumerate(grid_x):
                    for j, xi in enumerate(grid_y):
                        z_.append(np.array([[xi, yi]]))

                samples = sess.run(X_samples, feed_dict={z: np.squeeze(np.asarray(z_))})

                for idx, sample in enumerate(list(samples)):
                    floats_to_wav('out/{}.wav'.format(z_[idx]), sample.flatten(), 16000)

if __name__ == '__main__':
    check_tf_version()
    parser = set_config()
    (options, args) = parser.parse_args()
    if not options.run_inference:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        cur_dir = dir_path.split('/')[-1]
        script_name = os.path.basename(__file__).split('.')[0]
        options = extend_options(parser, cur_dir, script_name)

    train(options)
