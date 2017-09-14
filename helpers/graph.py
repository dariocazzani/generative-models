import tensorflow as tf

def get_variables(shape, scope):
	xavier = tf.contrib.layers.xavier_initializer()
	const = tf.constant_initializer(0.1)
	W = tf.get_variable('weight_{}'.format(scope), shape, initializer=xavier)
	b = tf.get_variable('bias_{}'.format(scope), shape[-1], initializer=const)
	return W, b

def linear(_input, output_dim, scope=None, reuse=None):
	with tf.variable_scope(scope, reuse=reuse):
		shape = [int(_input.get_shape()[1]), output_dim]
		W, b = get_variables(shape, scope)
		return tf.matmul(_input, W) + b

def AdamOptimizer(loss, lr, beta1, var_list=None, clip_grad=False):
	optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1)
	if not var_list:
		grads_and_vars = optimizer.compute_gradients(loss)
	else:
		grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
	if clip_grad:
		grads_and_vars = [(tf.clip_by_norm(grad, 1), var) for grad, var in grads_and_vars]
	train_op = optimizer.apply_gradients(grads_and_vars)
	return train_op, grads_and_vars

def clip_weights(vars, c, scope=None):
    with tf.variable_scope(scope):
        out = [var.assign(tf.clip_by_value(var, -c, c, name=var.name.split(':')[0])) for var in vars]
    return out

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
