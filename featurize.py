"""
This module extracts images stored in a directory titled 'masks', featurizes
and stores them in a csv file.

We define the labels:
    Fully covered = 0
    Not covered = 1
    Not a face = 2
    Partially covered = 3
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

DATA_PATH = os.path.join(os.getcwd(), "masks")
OUTPUT_FILENAME = os.path.join(os.getcwd(), "mask_data.csv")

# For storing the training data of each category before calling np.vstack()
arrs = []
# Information on how many of each label we have
counts = []

for index, directory in enumerate(os.listdir(DATA_PATH)):
    dirpath = os.path.join(DATA_PATH, directory)
    vectors = []
    files = os.listdir(dirpath)
    counts.append(len(files))

    # Assumes no subdirectories in the same directory as training data
    for filename in files:
        imgpath = os.path.join(dirpath, filename)
        img = Image.open(imgpath)
        feature_vector = np.array(img).flatten()
        vectors.append(feature_vector)

    # Convert to numpy array and add labels
    arr = np.array(vectors)
    labels = np.full((arr.shape[0], 1), index)
    data = np.hstack((arr, labels))

    # Add this category's data to arrs
    arrs.append(data)

# Combine data from all categories and write to output
output = np.vstack(arrs)
pd.DataFrame(output).to_csv(OUTPUT_FILENAME, header=None, index=None)

for index, count in enumerate(counts):
    print("The number of images with label {} is {}.".format(index, count))
