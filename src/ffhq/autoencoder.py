import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
import numpy as np
import matplotlib as mathplt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from batchup import data_source
from keras.models import Sequential
from keras.layers import InputLayer, Dense , Dropout, LeakyReLU, BatchNormalization, Reshape, Conv2DTranspose, Conv2D, UpSampling2D, Flatten, Activation, MaxPool2D
from keras.initializers import RandomNormal
from keras.preprocessing.image import ImageDataGenerator

batch_size = 16
epochs = 20
images_path = "../../dataset"

def rescale_img(img):
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) * 2
    return img

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

class Autoencoder(keras.Model):
  def build_encoder(self):
    model = Sequential()
    model.add(Conv2D(16, (5,5), padding="same"))
    model.add(LeakyReLU())
    model.add(MaxPool2D())
    model.add(Conv2D(25, (5,5), strides=(1,1), padding="same"))
    model.add(LeakyReLU()) 
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(16*16*25))
    model.add(LeakyReLU())
    model.add(Dense(16*16*3))
    model.add(LeakyReLU())
    model.add(Dense(self.latent_dim, activation="tanh"))
      
    return model

  def build_decoder(self):
    model = Sequential()
    model.add(Dense(16*16*3, input_dim=self.latent_dim))
    model.add(LeakyReLU())
    model.add(Dense(16*16*3))
    model.add(LeakyReLU())
    model.add(Dense(32*32*25))
    model.add(LeakyReLU())
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
    autoencoder = Autoencoder(491)
    autoencoder.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.5))
    generator = ImageDataGenerator(preprocessing_function=rescale_img)
    train_data = generator.flow_from_directory(images_path, target_size=(128, 128), batch_size=batch_size, class_mode=None)

    for iterations in range(epochs):
        batches = 0
        for batch in train_data:
            if(batches >= 70000 / batch_size):
                encoded_images = autoencoder.encoder.predict(batch)
                decoded_images = autoencoder.decoder.predict(encoded_images)
                plot(decoded_images)
                plt.savefig('out_autoencoder/{}.png'.format(str(iterations).zfill(3)), bbox_inches='tight')
                plt.close()
                print_iteration = False
                break
            loss = autoencoder.train_on_batch(batch, batch)
            print(loss)
            batches = batches + 1
        autoencoder.encoder.save_weights("encoder_full_491.h5")
        autoencoder.decoder.save_weights("decoder_full_491.h5")
