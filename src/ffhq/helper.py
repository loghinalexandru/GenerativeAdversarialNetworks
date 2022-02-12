import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import matplotlib as mathplt
import matplotlib.pyplot as plt 
import numpy as np
from autoencoder_conv import Autoencoder
images_path = "../../dataset"

encoder_dim = 245
latent_dim = 100
encoder_conv_weights = "encoder_conv.h5"
decoder_conv_weights = "decoder_conv.h5"
encoder_weights = "encoder_conv.h5"
decoder_weights = "decoder_conv.h5"
image = mathplt.image.imread("../../dataset/00000/00008.png")
image = np.reshape(image, (-1,128,128,3))

def rescale_img(img):
    img = (2  * np.array(img)) - 1
    return img

def random_sample(size):
    return np.random.normal(0., 1., size=[size,latent_dim])

def encoded_image(img, name):
    image = rescale_img(img)
    autoencoder = Autoencoder(encoder_dim)
    encoded_decoded_image = autoencoder.predict(image)
    autoencoder.encoder.load_weights(encoder_weights)
    autoencoder.decoder.load_weights(decoder_weights)
    autoencoder.encoder.trainable = False
    autoencoder.decoder.trainable = False
    encoded_decoded_image = autoencoder.predict(image)
    encoded_image = autoencoder.encoder.predict(image)
    encoded_image = (encoded_image + 1.) / 2.
    mathplt.image.imsave('encoded_image.png', np.reshape(encoded_image[0], (64,64)), cmap='binary')
    encoded_decoded_image = (encoded_decoded_image + 1.) / 2.
    mathplt.image.imsave(name, encoded_decoded_image[0])

def make_graph(input_file):
    file_handler = open(input_file , "r+")
    data = file_handler.readlines()
    data = [float(x.strip()) for x in data]
    data = data
    plt.xlabel("Iterations")
    plt.ylabel("MSE Loss")
    plt.yscale("log", base=10)
    plt.plot(data)
    plt.show()

def make_gan_graph(d_fake,d_real,gen):
    d_fake_handler = open(d_fake , "r+")
    d_real_handler = open(d_real , "r+")
    gen_handler = open(gen , "r+")
    data_d_fake = d_fake_handler.readlines()
    data_d_real = d_real_handler.readlines()
    data_gen = gen_handler.readlines()
    data_d_fake = [float(x.strip()) for x in data_d_fake]
    data_d_real = [float(x.strip()) for x in data_d_real]
    data_gen = [float(x.strip()) for x in data_gen]
    plt.xlabel("Iterations")
    plt.ylabel("MSE Loss")
    plt.plot(data_d_fake, label="d_fake")
    plt.plot(data_d_real, label="d_real")
    plt.yscale("log")
    plt.plot(data_gen, label="gen")
    plt.legend()
    plt.show()

def make_test_vector():
    test_vector = random_sample(1)[0]
    file_handler = open("test_vector.txt", "w+")
    for entry in test_vector:
        file_handler.write(str(entry) + "\n")


make_gan_graph("gan_classic_d_fake.txt", "gan_classic_d_real.txt", "gan_classic_gen.txt")