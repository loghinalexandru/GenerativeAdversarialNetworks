import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib as mathplt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from batchup import data_source
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense , Dropout, LeakyReLU, BatchNormalization, Reshape, Conv2DTranspose, Conv2D, UpSampling2D, Flatten, Activation, MaxPool2D
from tensorflow.keras.initializers import RandomNormal

batch_size = 16
max_input_size = 10000
epochs = 1000
images_path = "./dataset"

def load_data():
    input_data = []
    for entry in os.listdir(images_path):
        if(os.path.isdir(os.path.join(images_path, entry))):
            new_folder_path = os.path.join(images_path, entry)
        for image in os.listdir(os.path.join(images_path, entry)):
            if(len(input_data) == max_input_size):
                return input_data
            input_data.append(mathplt.image.imread(os.path.join(new_folder_path, image)))

    return input_data

def plot(samples):
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
        plt.imshow(sample)

    return fig

class Autoencoder(tf.keras.Model):
  def build_encoder(self):
    model = Sequential()
    model.add(Conv2D(16, (5,5), padding="same"))
    model.add(LeakyReLU())
    model.add(MaxPool2D())
    model.add(Dropout(0.25))
    model.add(Conv2D(25, (5,5), strides=(1,1), padding="same"))
    model.add(LeakyReLU()) 
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(16*16*25))
    model.add(LeakyReLU())
    model.add(Dropout(0.25))
    model.add(Dense(16*16*3))
    model.add(LeakyReLU())
    model.add(Dropout(0.25))
    model.add(Dense(self.latent_dim, activation="tanh"))
      
    return model

  def build_decoder(self):
    model = Sequential()
    model.add(Dense(16*16*3, input_dim=self.latent_dim))
    model.add(LeakyReLU())
    model.add(Dropout(0.25))
    model.add(Dense(16*16*3))
    model.add(LeakyReLU())
    model.add(Dense(32*32*25))
    model.add(LeakyReLU())
    model.add(Dropout(0.25))
    model.add(Reshape((32, 32, 25)))
    model.add(Conv2DTranspose(15, (5,5), padding='same', strides=(1,1)))
    model.add(LeakyReLU())
    model.add(UpSampling2D())
    model.add(Conv2DTranspose(3, (5,5), padding='same', strides=(1,1)))
    model.add(LeakyReLU())
    model.add(UpSampling2D())
    model.add(Conv2D(3, (5,5), padding='same', activation="tanh", strides=(1,1)))
    
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
    train_images = load_data()
    train_images = (2  * np.array(train_images)) - 1

    autoencoder = Autoencoder(128)
    autoencoder.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))
    for iterations in range(epochs):
        print_iteration = True
        batches = data_source.ArrayDataSource([train_images])
        for batch in batches.batch_iterator(batch_size, True):
            if(print_iteration):
                encoded_images = autoencoder.encoder(batch[0].reshape(batch_size,128,128,3))
                decoded_images = autoencoder.decoder(encoded_images)
                plot(decoded_images)
                plt.savefig('out/{}.png'.format(str(iterations).zfill(3)), bbox_inches='tight')
                plt.close()
                print_iteration = False
            loss = autoencoder.train_on_batch(batch[0].reshape(-1,128,128,3), batch[0].reshape(-1,128,128,3))
        print(loss)
        autoencoder.encoder.save_weights("encoder.h5")
        autoencoder.decoder.save_weights("decoder.h5")
