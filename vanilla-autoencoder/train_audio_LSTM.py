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

import subprocess
import tensorflow as tf
import numpy as np
import os
import glob

from helpers.data_dispatcher import CMajorScaleDistribution, NSynthGenerator

# Parameters
input_dim = 200
num_frames = 80
hidden_layer = 256

# The autoencoder network
def encoder(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.variable_scope('Encoder'):
        e_cell1 = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_layer)
        e_cell2 = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_layer)
        e_cells = tf.contrib.rnn.MultiRNNCell([e_cell1, e_cell2])
        outputs, _ = tf.contrib.rnn.static_rnn(e_cells, x, dtype=tf.float32)
        latent_variable = linear(outputs[-1], options.z_dim, 'e_latent_variable')
        return latent_variable

def decoder(z, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.variable_scope('Decoder'):
        rnn_input = []
        for _ in range(num_frames):
            rnn_input.append(z)
        d_cell1 = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_layer)
        d_cell2 = tf.contrib.rnn.LayerNormBasicLSTMCell(input_dim)
        d_cells = tf.contrib.rnn.MultiRNNCell([d_cell2])
        outputs, _ = tf.contrib.rnn.static_rnn(d_cells, rnn_input, dtype=tf.float32)
        logits = tf.stack(outputs, axis=1, name='logits')
        logits_reshaped = tf.reshape(logits, [-1, num_frames * input_dim])
        out = tf.nn.tanh(logits_reshaped)
        return out


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
            encoder_output = encoder(X_list)
            print('encoder_output: {}'.format(encoder_output.shape))
            decoder_output = decoder(encoder_output)

    with tf.variable_scope(tf.get_variable_scope()):
        X_samples = decoder(z, reuse=True)

    # Loss - MSE
    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.square(X - decoder_output))

    # Optimizer
    train_op, grads_and_vars = AdamOptimizer(loss, options.learning_rate, options.beta1)

    # Visualization
    tf.summary.scalar(name='Loss', tensor=loss)
    tf.summary.histogram(name='Encoder Distribution', values=encoder_output)

    for grad, var in grads_and_vars:
        tf.summary.histogram(var.name + '/gradient', grad)

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
                if not os.path.exists('out/'):
                    os.makedirs('out/')
                for i in range(options.epochs):

                    batch_x = data.__next__()
                    sess.run(train_op, feed_dict={X: batch_x})
                    if i % 10 == 0:
                        batch_loss, summary = sess.run([loss, summary_op], feed_dict={X: batch_x})
                        writer.add_summary(summary, global_step=step)
                        print("Epoch: {} - Loss: {}\n".format(i, batch_loss))
                        with open(options.logs_path + '/log.txt', 'a') as log:
                            log.write("Epoch: {} - Loss: {}\n".format(i, batch_loss))

                        samples = sess.run(X_samples, feed_dict={z: np.random.randn(1, options.z_dim)})
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
            print('Restoring latest saved TensorFlow model...')
            raise NotImplementedError("Todo")

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
