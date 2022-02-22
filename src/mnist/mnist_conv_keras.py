import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
import numpy as np
import matplotlib as mathplt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from batchup import data_source
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import InputLayer, Dense , Dropout, LeakyReLU, BatchNormalization, Reshape, Conv2DTranspose, Conv2D, Flatten, Activation
from keras import activations

latent_dim = 100
batch_size = 128
epochs = 100

def random_sample(size):
    return np.random.normal(0, 1., size=[size,latent_dim])

def plot(samples):
    # Rescale to [0,1] from [-1,1]
    samples = (samples + 1.) / 2.
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28,28), cmap='Greys_r')

    return fig

def plot_single(sample):
    # Rescale to [0,1] from [-1,1]
    sample = (sample + 1.) / 2.
    plt.figure(figsize=(0.37,0.37))
    plt.imshow(sample, cmap='Greys_r')
    plt.axis('off')

def build_generator():
    model = Sequential()
    model.add(Dense(7*7*256, use_bias=False, input_dim=latent_dim))
    model.add(LeakyReLU())
    model.add(Reshape((7, 7, 256)))
    model.add(Conv2DTranspose(128, (3,3), use_bias=False, padding='same', strides=(1,1)))
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(64, (3,3), use_bias=False, padding='same', strides=(2,2)))
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(1, (3,3), use_bias=False, padding='same', strides=(2,2), activation='tanh'))

    return model

def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, (5,5), strides=(2,2), use_bias=False, padding='same',  input_shape=[28,28,1]))
    model.add(LeakyReLU())
    model.add(Conv2D(128, (5,5), strides=(2,2),  use_bias=False,  padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(128, (5,5), strides=(2,2), use_bias=False, padding='same'))
    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.5))

    return model

def build_gan_model(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.5))

    return model

if __name__ == "__main__":
    # load_data()

    # Map to [-1,1]
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data(path="mnist.npz")
    train_images = (2  * np.array(train_images) / 255) - 1

    i = 0
    generator = build_generator()
    discriminator = build_discriminator()
    gan_model = build_gan_model(generator, discriminator)

    batches = data_source.ArrayDataSource([train_images], repeats=epochs)

    for batch in batches.batch_iterator(batch_size, True):
        if i % 100 == 0:
            samples = generator.predict(random_sample(1))
            print(np.shape(samples))
            fig = plot_single(samples[0])
            plt.savefig('out_classic_single/{}.png'.format(str(i).zfill(3)), bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close(fig)
        
        real_data_input, real_data_label = batch[0].reshape(-1,28,28,1) , np.repeat(np.random.uniform(0.3,0.7), batch_size)
        fake_data_input, fake_data_label = generator.predict(random_sample(batch_size)), np.repeat(np.random.uniform(-0.7,-0.3), batch_size)

        d_real_loss = discriminator.train_on_batch(real_data_input, real_data_label)
        d_fake_loss = discriminator.train_on_batch(fake_data_input, fake_data_label)

        gan_loss = gan_model.train_on_batch(random_sample(batch_size), np.repeat(np.random.uniform(0.3,0.7), batch_size))
        print(d_real_loss + d_fake_loss, gan_loss)
        i = i + 1