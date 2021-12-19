import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
import numpy as np
import matplotlib as mathplt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.initializers import RandomNormal
from keras.preprocessing.image import ImageDataGenerator
from autoencoder import Autoencoder

latent_dim = 10
batch_size = 16
encoder_dim = 245
epochs = 100
images_path = "../../dataset"
init = RandomNormal(stddev=0.02)

def rescale_img(img):
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) * 2
    return img

def random_sample(size):
    return np.random.normal(0., 1., size=[size,latent_dim])

def plot(samples):
    # Rescale to [0,1] from [-1,1]
    samples = (samples + 1.) / 2.
    samples = np.array(samples)
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
    model.add(Dense(64, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(encoder_dim, activation='tanh'))

    return model

def build_discriminator():
    model = Sequential()
    model.add(Dense(256, input_dim=encoder_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

    return model

def build_gan_model(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

    return model

if __name__ == "__main__":
    autoencoder = Autoencoder(encoder_dim)
    datagen = ImageDataGenerator(preprocessing_function=rescale_img)
    train_data = datagen.flow_from_directory(images_path, target_size=(128, 128), batch_size=batch_size, class_mode=None)

    autoencoder.predict(np.reshape(train_data.next(), (-1,128,128,3)))
    autoencoder.encoder.load_weights("encoder_full_245.h5")
    autoencoder.decoder.load_weights("decoder_full_245.h5")
    autoencoder.encoder.trainable = False
    autoencoder.decoder.trainable = False

    generator = build_generator()
    discriminator = build_discriminator()
    gan_model = build_gan_model(generator, discriminator)

    for iteration in range(epochs):
        batches = 0
        for batch in train_data:
            batches = batches + 1
            real_data_input, real_data_label = autoencoder.encoder.predict(batch), np.repeat(np.random.uniform(0.8, 1.0), batch_size)
            fake_data_input, fake_data_label = generator.predict(random_sample(batch_size)), np.repeat(np.random.uniform(-0.1, 0.1), batch_size)

            d_real_loss = discriminator.train_on_batch(real_data_input, real_data_label)
            d_fake_loss = discriminator.train_on_batch(fake_data_input, fake_data_label)

            gan_loss = gan_model.train_on_batch(random_sample(batch_size), np.ones(batch_size))
            print(d_real_loss, d_fake_loss, gan_loss)
            
            if(batches >= 70000 / batch_size):
                break

        samples = np.array(generator.predict(random_sample(16)))
        fig = plot(autoencoder.decoder.predict(samples))
        plt.savefig('out/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
        plt.close(fig)