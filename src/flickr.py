import tensorflow as tf
import os
import keras
import numpy as np
import matplotlib as mathplt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from batchup import data_source
from keras.models import Sequential
from keras.layers import InputLayer, Dense , Dropout, LeakyReLU, BatchNormalization, Reshape, Conv2DTranspose, Conv2D, Flatten, Activation
from keras import activations

latent_dim = 100
max_input_size = 10000
batch_size = 18
epochs = 100
images_path = "../dataset"
input_data = []

def load_data():
    for entry in os.listdir(images_path):
        if(os.path.isdir(os.path.join(images_path, entry))):
            new_folder_path = os.path.join(images_path, entry)
        for image in os.listdir(os.path.join(images_path, entry)):
            if(len(input_data) == max_input_size):
                return
            input_data.append(mathplt.image.imread(os.path.join(new_folder_path, image)))

def random_sample(size):
    return np.random.normal(-1., 1., size=[size,latent_dim])

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
        plt.imshow(sample.reshape(128,128,3))

    return fig

def build_generator():
    model = Sequential()
    model.add(Dense(32*32*256, input_dim=latent_dim))
    model.add(Reshape((32, 32, 256)))
    model.add(Conv2DTranspose(128, (5,5), padding='same', strides=(1,1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2DTranspose(64, (5,5), padding='same', strides=(2,2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2DTranspose(3, (5,5), padding='same', strides=(2,2)))
    model.add(Activation(activations.tanh))

    return model

def build_discriminator():
    model = Sequential()
    model.add(Conv2D(256, (5,5), padding='same',  input_shape=[128,128,3]))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(128, (5,5), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

    return model

def build_gan_model(generator, discriminator):
    for layer in discriminator.layers:
        layer.trainable = False

    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

    return model

if __name__ == "__main__":
    load_data()

    # Map to [-1,1]
    train_images = (2  * np.array(input_data) / 255) - 1

    i = 0
    generator = build_generator()
    discriminator = build_discriminator()
    gan_model = build_gan_model(generator, discriminator)

    batches = data_source.ArrayDataSource([np.array(train_images)], repeats=epochs)

    for batch in batches.batch_iterator(batch_size, True):
        samples = generator.predict(random_sample(16))
        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        plt.close(fig)
        
        real_data_input, real_data_label = batch[0] , np.ones(batch_size)
        fake_data_input, fake_data_label = generator.predict(random_sample(batch_size)), np.zeros(batch_size)

        d_real_loss = discriminator.train_on_batch(real_data_input, real_data_label)
        d_fake_loss = discriminator.train_on_batch(fake_data_input, fake_data_label)

        gan_loss = gan_model.train_on_batch(random_sample(batch_size), np.ones(batch_size))
        print(d_real_loss, d_fake_loss, gan_loss)
        i = i + 1