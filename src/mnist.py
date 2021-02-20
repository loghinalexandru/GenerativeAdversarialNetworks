import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

iterations = 100000
batch_size = 128
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
real_data = tf.placeholder(tf.float32, shape = [None, 784])
fake_data = tf.placeholder(tf.float32, shape = [None, 100])

def random_sample():
    return np.random.uniform(-1., 1., size=[batch_size,100])

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
    loss_generator = -tf.reduce_mean(tf.log(1 - result_fake))

    discriminator_solver = tf.train.AdamOptimizer().minimize(loss_discriminator)
    generator_solver = tf.train.AdamOptimizer().minimize(loss_generator)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    train_images = (train_images.astype(np.float32) - 127.5) / 127.5
    train_images = [np.reshape(np.array(val), (1,784)) for val in train_images]
    print(np.shape(train_images))
    print(np.shape(random_sample()))

    for i in range(iterations):
        print(np.shape(train_images))
        batch = train_images[i * batch_size : (i + 1) * batch_size]
        _, discriminator_curr = sess.run([discriminator_solver, loss_discriminator], feed_dict={real_data: batch, fake_data: random_sample()})
        _, generator_curr = sess.run([generator_solver, loss_generator], feed_dict={fake_data : random_sample()})
        print('D loss: {:.4}'. format(discriminator_curr))