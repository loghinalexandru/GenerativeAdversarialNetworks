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

latent_dim = 3
batch_size = 128
epochs = 10

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
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

class Autoencoder(tf.keras.Model):

  def build_encoder(self):
      model = Sequential()
      model.add(Conv2D(16, (3,3), padding='same', strides=(2,2), input_shape=[28,28,1]))
      model.add(LeakyReLU())
      model.add(Conv2D(16, (3,3), padding='same', strides=(1,1)))
      model.add(LeakyReLU())
      model.add(Conv2D(1, (3,3), padding='same', activation='tanh', strides=(1,1)))
      
      return model

  def build_decoder(self):
      model = Sequential()
      model.add(Conv2DTranspose(16, (3,3), strides=(2,2), padding='same'))
      model.add(LeakyReLU())
      model.add(Conv2DTranspose(32, (3,3), strides=(1,1),  padding='same'))
      model.add(LeakyReLU())
      model.add(Conv2D(1, (3,3), activation='tanh', padding='same'))
      
      return model

  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = self.build_encoder()
    self.decoder = self.build_decoder()

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
    train_images = (2  * np.array(train_images) / 255) - 1

    autoencoder = Autoencoder(latent_dim)
    autoencoder.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

    i = 0
    for iterations in range(epochs):
        batches = data_source.ArrayDataSource([train_images])
        for batch in batches.batch_iterator(batch_size, True):
            if(i % 100 == 0):
                encoded_image = autoencoder.encoder(batch[0][0].reshape(-1,28,28,1))
                print(np.shape(encoded_image))
                decoded_image = autoencoder.decoder(encoded_image)
                plt.imshow(np.reshape(decoded_image, (28,28,1)), cmap='Greys_r')
                plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                plt.close()
            i = i + 1
            loss = autoencoder.train_on_batch(batch[0].reshape(-1,28,28,1), batch[0].reshape(-1,28,28,1))
            print(loss)
        autoencoder.encoder.save_weights("encoder.h5")
        autoencoder.decoder.save_weights("decoder.h5")