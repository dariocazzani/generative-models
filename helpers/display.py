import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot(sess, z, X_samples, num_images, height, width):
    samples = []
    grid_x = np.linspace(-2, 2, num_images)
    grid_y = np.linspace(-2, 2, num_images)
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
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
