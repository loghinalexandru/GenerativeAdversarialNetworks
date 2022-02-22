import numpy as np
import os
import matplotlib as mathplt
import matplotlib.pyplot as plt
import argparse



def calculateDistance(i1, i2):
    return np.sum(np.abs(i1-i2))

images_path = "../../dataset"
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True)
args = vars(ap.parse_args())

query_image = mathplt.image.imread(args["query"])[:,:,:3]
distances_vector = []

for root, dirs, files in os.walk(images_path):
     for file in files:
         image =  mathplt.image.imread(os.path.join(root, file), "r")
         distances_vector.append((calculateDistance(query_image, image), file))
         print(file)

print(sorted(distances_vector, key=lambda x : x[0])[:5])