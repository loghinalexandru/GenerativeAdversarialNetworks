import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from batchup import data_source


iterations = 10000
batch_size = 120
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
real_data = tf.placeholder(tf.float32, shape = [None, 784])
fake_data = tf.placeholder(tf.float32, shape = [None, 100])
test_data = tf.placeholder(tf.float32, shape = [None, 100])

def random_sample(size):
    return np.random.uniform(-1., 1., size=[size,100])

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def generator_network(x):
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        x = tf.layers.dense(x, 128, activation=tf.nn.relu)
        x = tf.layers.dense(x, 784)
        x = tf.nn.sigmoid(x)
    return x

def discriminator_network(x):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        x = tf.layers.dense(x, 128, activation=tf.nn.relu)
        x = tf.layers.dense(x, 1)
        x = tf.nn.sigmoid(x)
    return x

if __name__ == "__main__":
    result_real = discriminator_network(real_data)
    result_fake = discriminator_network(generator_network(fake_data))

    loss_discriminator = -tf.reduce_mean(tf.log(result_real) + tf.log(1 - result_fake))
    loss_generator = -tf.reduce_mean(tf.log(result_fake))

    discriminator_solver = tf.train.AdamOptimizer().minimize(loss_discriminator)
    generator_solver = tf.train.AdamOptimizer().minimize(loss_generator)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    train_images = (train_images.astype(np.float32) - 127.5) / 127.5
    train_images = [np.reshape(np.array(val), (1,784)).flatten() for val in train_images]
    batches = data_source.ArrayDataSource([np.array(train_images)], repeats=iterations)
    i = 0
    for batch in batches.batch_iterator(batch_size, False):
        if i % 1000 == 0:
            samples = sess.run(generator_network(test_data), feed_dict={test_data : random_sample(16)})
            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            plt.close(fig)

        _, discriminator_curr = sess.run([discriminator_solver, loss_discriminator], feed_dict={real_data: batch[0], fake_data: random_sample(batch_size)})
        _, generator_curr = sess.run([generator_solver, loss_generator], feed_dict={fake_data : random_sample(batch_size)})
        print('D loss: {:.4}'. format(discriminator_curr))
        i=i+1
        