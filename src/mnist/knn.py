import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
from keras.datasets import mnist
import matplotlib as mathplt
import matplotlib.pyplot as plt
import argparse

def calculateDistance(i1, i2):
    return np.sum(np.abs(i1-i2))

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

images_path = "../../dataset"
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True, help="Directory of the image that will be compared")
args = vars(ap.parse_args())

query_image = mathplt.image.imread(args["query"])[:,:,:3]
query_image = rgb2gray(query_image)   
distances_vector = []

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

for file in train_images:
    norm = file/255
    distances_vector.append((calculateDistance(query_image, norm), norm))

result = sorted(distances_vector, key=lambda x : x[0])[:5]
result.append(("query", query_image))

for i,entry in enumerate(result):
    plt.imshow(entry[1], cmap='Greys_r')
    plt.axis('off')
    plt.savefig('{}.png'.format(str(i).zfill(3)), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()