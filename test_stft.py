import tensorflow as tf
import numpy as np


x = tf.placeholder(tf.float32, shape=[2, 1000])
stft = tf.contrib.signal.stft(x, 100, 50, fft_length=512)
magnitude = tf.abs(stft)
unstacked = tf.unstack(magnitude, num=None, axis=1)
istft = tf.contrib.signal.inverse_stft(stft, 100, 50, fft_length=512)

with tf.Session() as sess:
    _input = np.random.randn(2, 1000)
    stft_value, istft_value, magnitude_value, unstacked_value = sess.run([stft, istft, magnitude, unstacked], feed_dict={x: _input})
    print(len(unstacked_value))
    print(stft_value.shape)
    print(istft_value.shape)
    print(magnitude_value.shape)
    print(np.mean(np.abs(_input - istft_value)))
