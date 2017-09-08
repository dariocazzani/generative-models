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
from helpers.display import plot

import subprocess
import tensorflow as tf
import numpy as np
import os
import glob

from tensorflow.examples.tutorials.mnist import input_data

# Get the MNIST data
mnist = input_data.read_data_sets('../Data', one_hot=True)

# Parameters
"""
TO ADD TO OPTIONS.. MAYBE
"""
input_dim = 28
hidden_layer = 1000
sequence_length = 28*2
teacher_forcing = False

# Q(z|X)
def encoder(x, reuse=False):
	if reuse:
		tf.get_variable_scope().reuse_variables()
	with tf.variable_scope('Encoder'):
		e_cell1 = tf.nn.rnn_cell.LSTMCell(hidden_layer)
		cells = tf.contrib.rnn.MultiRNNCell([e_cell1])
		_, states = tf.contrib.rnn.static_rnn(cells, x, dtype=tf.float32)
		state_c, state_h = states[-1]
		hidden_state = tf.concat([state_c, state_h], 1)
		z_mu = linear(hidden_state, options.z_dim, 'z_mu')
		z_logvar = linear(hidden_state, options.z_dim, 'z_logvar')
		return z_mu, z_logvar

def sample_z(mu, log_var):
	eps = tf.random_normal(shape=tf.shape(mu))
	return mu + tf.exp(log_var / 2) * eps

def rnn_forward_pass(cells, _input, states):
	cell_outputs = []
	cell_states = []
	assert(len(cells) == len(states))
	num_layers = len(cells)
	for layer in range(num_layers):
		with tf.variable_scope('layer_{}'.format(layer)):
			if layer == 0:
				o, s = cells[layer](_input, states[0])
			else:
				o, s = cells[layer](cell_outputs[layer-1], states[layer])
			cell_outputs.append(o)
			cell_states.append(s)
	return cell_outputs[-1], cell_states

def seq2seq(cells, initial_states, start, sequence_length, inputs=None):
	outputs = []
	states = []
	for step in range(sequence_length):
		if step == 0:
			o, s = rnn_forward_pass(cells, start, initial_states)
		else:
			previous_output = inputs[step-1] if inputs else outputs[step-1]
			previous_states = states[step-1]
			o, s = rnn_forward_pass(cells, previous_output, previous_states)
		outputs.append(o)
		states.append(s)
	# return states for last step only
	return outputs, states[-1]

# P(X|z)
def decoder(z, inputs=None, reuse=False, sequence_length=input_dim):
	if reuse:
		tf.get_variable_scope().reuse_variables()
	with tf.variable_scope('Decoder'):
		lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(hidden_layer)
		lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(input_dim)
		expanded_z_1 = tf.nn.relu(linear(z, hidden_layer, 'linear_expand_z_1'))
		expanded_z_2 = tf.nn.relu(linear(z, input_dim, 'linear_expand_z_3'))
		initial_state_1 = (expanded_z_1, expanded_z_1)
		initial_state_2 = (expanded_z_2, expanded_z_2)
		first_input = tf.add(tf.zeros_like(expanded_z_1[:, :input_dim]), 1.)

		outputs, states = seq2seq([lstm_cell_1, lstm_cell_2],
								 [initial_state_1, initial_state_2],
								 first_input,
								 sequence_length,
								 inputs=inputs)

		# logits = tf.add_n(outputs)

		logits = tf.stack(outputs, axis=1, name='logits')
		logits_reshaped = tf.reshape(logits, [-1, input_dim * sequence_length])
		out = tf.nn.sigmoid(logits_reshaped)
		return out, logits_reshaped

def train(options):
	# Placeholders for input data and the targets
	with tf.name_scope('Input'):
		X = tf.placeholder(dtype=tf.float32, shape=[None, input_dim * input_dim], name='Input')
		input_images = tf.reshape(X, [-1, input_dim, input_dim, 1])
		X_reshaped = tf.reshape(X, [-1, input_dim, input_dim])
		# Unstack to get a list of vertical slices of shape (batch_size, input_dim)
		X_list = tf.unstack(X_reshaped, num=None, axis=1)

	with tf.name_scope('Latent_variable'):
		z = tf.placeholder(dtype=tf.float32, shape=[None, options.z_dim], name='Latent_variable')

	with tf.name_scope('Autoencoder'):
		with tf.variable_scope(tf.get_variable_scope()):
			z_mu, z_logvar = encoder(X_list)
			z_sample = sample_z(z_mu, z_logvar)
			decoder_output, logits = decoder(z_sample, inputs=X_list if teacher_forcing else None)
			generated_images = tf.reshape(decoder_output, [-1, input_dim, input_dim, 1])

	with tf.variable_scope(tf.get_variable_scope()):
		X_samples, _ = decoder(z, reuse=True, sequence_length=sequence_length)

	# Loss
	with tf.name_scope('Loss'):
		# E[log P(X|z)]
		reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)
		# D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
		kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
		# VAE loss
		vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

	# Optimizer
	train_op, grads_and_vars = AdamOptimizer(vae_loss, options.learning_rate, options.beta1)

	# Visualization
	tf.summary.scalar(name='Loss', tensor=vae_loss)
	tf.summary.histogram(name='Sampled variable', values=z_sample)

	for grad, var in grads_and_vars:
		tf.summary.histogram('Gradients/' + var.name, grad)
		tf.summary.histogram('Values/' + var.name, var)

	tf.summary.image(name='Input Images', tensor=input_images, max_outputs=10)
	tf.summary.image(name='Generated Images', tensor=generated_images, max_outputs=10)
	summary_op = tf.summary.merge_all()

	saver = tf.train.Saver()
	step = 0
	init = tf.global_variables_initializer()
	n_batches = int(mnist.train.num_examples / options.batch_size)
	with tf.Session() as sess:
		sess.run(init)
		if not options.run_inference:
			try:
				writer = tf.summary.FileWriter(logdir=options.tensorboard_path, graph=sess.graph)
				for epoch in range(options.epochs):
					for iteration in range(n_batches):
						batch_x, _ = mnist.train.next_batch(options.batch_size)

						# Train
						sess.run(train_op, feed_dict={X: batch_x})

						if iteration % 50 == 0:
							batch_loss, summary = sess.run([vae_loss, summary_op], feed_dict={X: batch_x})
							writer.add_summary(summary, global_step=step)
							print("Epoch: {} - Iteration {} - Loss: {:.4f}\n".format(epoch, iteration, batch_loss))

						step += 1
					saver.save(sess, save_path=options.checkpoints_path, global_step=step)

				print("Model Trained!")
			except KeyboardInterrupt:
				print('Stopping training...')
				saver.save(sess, save_path=options.checkpoints_path, global_step=step)
		else:
			print('Restoring latest saved TensorFlow model...')
			dir_path = os.path.dirname(os.path.realpath(__file__))
			cur_dir = dir_path.split('/')[-1]
			script_name = os.path.basename(__file__).split('.')[0]
			experiments = glob.glob(os.path.join(options.MAIN_PATH, cur_dir) + '/{}*'.format(script_name))
			experiments.sort(key=lambda x: os.path.getmtime(x))
			if len(experiments) > 0:
				print('Restoring: {}'.format(experiments[-1]))
				saver.restore(sess, tf.train.latest_checkpoint(os.path.join(experiments[-1], 'checkpoints')))
				plot(sess, z, X_samples, num_images=5, height=sequence_length, width=input_dim)

			else:
				print('No checkpoint found at {}'.format(os.path.join(options.MAIN_PATH, cur_dir)))

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
