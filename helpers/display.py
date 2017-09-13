import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot(sess, z, X_samples, num_images, height, width, condition=False, c=None):
    samples = []
    num_classes = 10
    if condition:
        y = np.zeros(shape=[1, num_classes])
        digit = input('What digit do you want to generate? [0-9]: ')
        digit2 = input('What digit do you want to combine the previous one? [0-9]: ')

        digit = int(digit)
        digit2 = int(digit2)
        try:
            y[:, digit] = 1.
            y[:, digit2] = 1.
        except Exception as e:
            print('Could not generate condition: {}'.format(e))

    grid_x = np.linspace(-2, 2, num_images)
    grid_y = np.linspace(-2, 2, num_images)
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            if condition:
                samples.append(sess.run(X_samples, feed_dict={z: z_sample, c: y}))
            else:
                samples.append(sess.run(X_samples, feed_dict={z: z_sample}))

    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(num_images, num_images)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(height, width), cmap='Greys_r')
    plt.show()
    # plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
    plt.close(fig)
