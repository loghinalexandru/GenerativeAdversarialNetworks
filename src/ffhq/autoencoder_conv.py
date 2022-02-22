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

batch_size = 32
epochs = 100
images_path = "../../dataset"
encoder_weights = "encoder_conv.h5"
decoder_weights = "decoder_conv.h5"
output_folder = "out_autoencoder_conv_new"


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
    model.add(Conv2D(96, (11,11), padding="same", strides=(2,2)))
    model.add(LeakyReLU())
    model.add(Conv2D(128, (5,5), strides=(2,2), padding="same"))
    model.add(LeakyReLU())
    model.add(Conv2D(128, (3,3), strides=(2,2), padding="same"))
    model.add(LeakyReLU())
    model.add(Conv2D(16, (3,3), strides=(1,1), padding="same", activation="tanh"))
    
    return model

  def build_decoder(self):
    model = Sequential()
    model.add(Conv2DTranspose(128, (3,3), strides=(2,2), padding="same"))
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(96, (3,3), strides=(2,2), padding="same"))
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(3, (3,3), padding='same', activation="tanh", strides=(2,2)))
    
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
    mse_loss = open(r"autoencoder_mse_loss.txt", "w+")
    autoencoder = Autoencoder(245)
    autoencoder.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.5))
    generator = ImageDataGenerator(preprocessing_function=rescale_img)
    train_data = generator.flow_from_directory(images_path, target_size=(128, 128), batch_size=batch_size, class_mode=None)

    # if(os.path.exists(encoder_weights) and os.path.exists(decoder_weights)):
    #   autoencoder.predict(np.reshape(train_data.next(), (-1,128,128,3)))
    #   autoencoder.encoder.load_weights(encoder_weights)
    #   autoencoder.decoder.load_weights(decoder_weights)

    for iterations in range(epochs):
        batches = 0
        for batch in train_data:
            if(batches >= train_data.samples / batch_size):
                encoded_images = autoencoder.encoder.predict(batch[:16])
                decoded_images = autoencoder.decoder.predict(encoded_images)
                plot(decoded_images)
                plt.savefig(output_folder + '/{}.png'.format(str(iterations).zfill(3)), bbox_inches='tight')
                plt.close()
                print_iteration = False
                break
            loss = autoencoder.train_on_batch(batch, batch)
            mse_loss.write(str(loss) + '\n')
            batches = batches + 1
        # autoencoder.encoder.save_weights(encoder_weights)
        # autoencoder.decoder.save_weights(decoder_weights)
