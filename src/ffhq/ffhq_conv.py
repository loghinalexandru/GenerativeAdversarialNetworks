import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
import numpy as np
import matplotlib as mathplt
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Reshape, Flatten, Conv2DTranspose,Conv2D
from keras.initializers import RandomNormal
from keras.preprocessing.image import ImageDataGenerator
from autoencoder_conv import Autoencoder

latent_dim = 100
batch_size = 16
encoder_dim = 4096
epochs = 100
images_path = "../../dataset"
encoder_weights = "encoder_conv.h5"
decoder_weights = "decoder_conv.h5"
generator_weights = "generator_autoencoder.h5"
discriminator_weights = "discriminator_autoencoder.h5"
output_folder = "out_conv_gan_test"
init = RandomNormal(stddev=0.02)

def rescale_img(img):
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) * 2
    return img

def random_sample(size):
    return np.random.normal(0., 1., size=[size,latent_dim])

def generate_image_from_vector(input_file):
    file_handler = open(input_file , "r+")
    data = file_handler.readlines()
    test_vector = [float(x.strip()) for x in data]
    test_vector  = np.reshape(test_vector, (1,100))
    generated_code = generator.predict(test_vector)
    result_image = autoencoder.decoder.predict(generated_code)[0]
    result_image = (result_image + 1.) / 2.
    mathplt.image.imsave("test_vector.png", result_image)

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
    model.add(Dense(16*16*32, input_dim=latent_dim))
    model.add(LeakyReLU())
    model.add(Reshape((16, 16, 32)))
    model.add(Conv2DTranspose(256, (3,3), padding='same', strides=(1,1)))
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(128, (3,3), padding='same', strides=(1,1)))
    model.add(LeakyReLU())
    model.add(Conv2D(16, (3,3), padding='same', strides=(1,1), activation='tanh'))

    return model

def build_discriminator():
    model = Sequential()
    model.add(Conv2D(256, (3,3), strides=(1,1), padding='same',  input_shape=[16,16,16]))
    model.add(LeakyReLU())
    model.add(Conv2D(128, (3,3), strides=(1,1), padding='same'))
    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
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
    d_real_file = open(r"gan_hybrid_d_real.txt", "w+")
    d_fake_file = open(r"gan_hybrid_d_fake.txt", "w+")
    gen_file = open(r"gan_hybrid_gen.txt", "w+")
    autoencoder = Autoencoder(encoder_dim)
    datagen = ImageDataGenerator(preprocessing_function=rescale_img)
    train_data = datagen.flow_from_directory(images_path, target_size=(128, 128), batch_size=batch_size, class_mode=None)

    autoencoder.predict(np.reshape(train_data.next(), (-1,128,128,3)))
    autoencoder.encoder.load_weights(encoder_weights)
    autoencoder.decoder.load_weights(decoder_weights)
    autoencoder.encoder.trainable = False
    autoencoder.decoder.trainable = False

    generator = build_generator()
    discriminator = build_discriminator()
    gan_model = build_gan_model(generator, discriminator)

    # if(os.path.exists(generator_weights) and os.path.exists(discriminator_weights)):
    #     generator.load_weights(generator_weights)
    #     discriminator.load_weights(discriminator_weights)

    generate_image_from_vector("test_vector.txt")

    for iteration in range(epochs):
        batches = 0
        for batch in train_data:
            batches = batches + 1
            real_data_input, real_data_label = autoencoder.encoder.predict(batch), np.repeat(np.random.uniform(0.8, 1.0), len(batch))
            fake_data_input, fake_data_label = generator.predict(random_sample(len(batch))), np.repeat(np.random.uniform(-0.1, 0.1), len(batch))

            d_real_loss = discriminator.train_on_batch(real_data_input, real_data_label)
            d_fake_loss = discriminator.train_on_batch(fake_data_input, fake_data_label)

            d_real_file.write(str(d_real_loss) + "\n")
            d_fake_file.write(str(d_fake_loss) + "\n")

            gan_loss = gan_model.train_on_batch(random_sample(len(batch)), np.ones(len(batch)))
            print(d_real_loss, d_fake_loss, gan_loss)
            gen_file.write(str(gan_loss) + "\n")

            if(batches >= train_data.samples / batch_size):
                break

        # generator.save_weights(generator_weights)
        # discriminator.save_weights(discriminator_weights)
        samples = np.array(generator.predict(random_sample(16)))
        fig = plot(autoencoder.decoder.predict(samples))
        plt.savefig(output_folder + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
        plt.close(fig)