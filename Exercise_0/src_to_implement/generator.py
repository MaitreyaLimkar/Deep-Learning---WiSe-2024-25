import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import random
import skimage

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        # Class dictionary implemented
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

        self.file_path = file_path      # File Path: String
        self.label_path = label_path    # Label Path: String
        self.batch_size = batch_size    # Batch Size: Integer
        self.image_size = image_size    # Image Size: List of Integers [height, width, channel]
        self.rotation = rotation        # Rotation Bool
        self.mirroring = mirroring      # Mirroring Bool
        self.shuffle = shuffle          # Shuffle Bool

        self.batch_start = 0
        self.epoch_num = 0

        # Loading the labels from the JSON file
        with open(self.label_path, 'r') as file:
            self.labels = json.load(file)

        # Loading images and resizing
        self.images = []
        for i in sorted(self.labels, key=lambda x: int(x)):
            img = np.load(self.file_path + str(i) + ".npy")
            img = skimage.transform.resize(img, self.image_size)
            self.images.append(img)

        self.labels = [self.labels[key] for key in sorted(self.labels, key=lambda x: int(x))]

        if self.shuffle:
            self.shuffle_data()

    def shuffle_data(self):
        combined_list = list(zip(self.images, self.labels))
        random.shuffle(combined_list)
        self.images, self.labels = zip(*combined_list)

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases

        if self.batch_start >= len(self.images):
            self.batch_start = 0
            self.epoch_num += 1
            if self.shuffle:
                self.shuffle_data()

        end = self.batch_start + self.batch_size
        batch_images = self.images[self.batch_start:end]
        batch_labels = self.labels[self.batch_start:end]

        # Now at the end of epoch, if the batch is smaller than batch_size, we incorporate images from the start
        if len(batch_images) < self.batch_size:
            additional_images = self.images[:self.batch_size - len(batch_images)]
            additional_labels = self.labels[:self.batch_size - len(batch_labels)]
            batch_images.extend(additional_images)
            batch_labels.extend(additional_labels)

        self.batch_start = end
        augmented_images = [self.augment(img) for img in batch_images]

        return np.array(augmented_images), np.array(batch_labels)

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image

        if self.rotation:
            img = np.rot90(img, random.randint(0,3), axes=(0,1)) ## added case to no rot
        if self.mirroring:
            img = np.flip(img,axis=0)

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch_num

    def class_name(self, x):
        # This function returns the class name for a specific input
        return self.class_dict[x]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        figures_in_x = self.batch_size // 3
        figures_in_y = self.batch_size // figures_in_x
        images, labels = self.next()

        # fig = plt.figure()
        loc, ax = plt.subplots(figures_in_x, figures_in_y)
        for i in range(figures_in_x):
            for j in range(figures_in_y):
                ax[i, j].imshow(images[i * figures_in_y + j])
                ax[i, j].set_title(self.class_name(labels[i * figures_in_y + j]))
                ax[i, j].axis('off')
        plt.show()