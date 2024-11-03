"""This file is used to implement the classes Checker and Circle resolpectively.
    We will implement them using the constructor method __init__(), a method draw() to create the pattern,
    and finally a visualization function show()."""

import numpy as np
import matplotlib.pyplot as plt
from keras.src.legacy.backend import dtype


class Checker:
  def __init__(self, resol, ts):
    """ resol (int) = Number of pixels in each dimension
        ts (int)  = Number of pixels individual tiles has in each direction """
    self.resol = resol
    self.ts = ts
    self.output = None

  def draw(self):
    if self.resol % (2 * self.ts) != 0:  # To avoid truncation of the checkerboard
        raise ValueError("Error! The resolution must be divisible by two.")

    checker_pat = np.ones((2*self.ts,2*self.ts),dtype=int)

    # To make sure that the top left and the bottom right of the array is zero.
    checker_pat[:self.ts,:self.ts] = 0
    checker_pat[self.ts:,self.ts:] = 0

    # For the number of tiles in every direction. It should be an integer so we use '//'.
    tiles = self.resol // (2 * self.ts)

    # We tile up our array to the output.
    self.output = np.tile(checker_pat,(tiles, tiles))
    return self.output.copy()

  def show(self):
    plt.imshow(self.output,cmap='gray')
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    plt.show()


class Circle:
    def __init__(self, resol, rad, pos):
        """ resol (int) = Number of pixels in each dimension
            rad (int)   = Radius of the circle
            pos (tuple) = Position of the circle in x and y direction """
        self.resol = resol
        self.rad = rad
        self.pos = pos
        self.output = np.zeros((self.resol,self.resol),dtype=int)

    def draw(self):
        # We create a 2D coordinate array for pixel positions
        y, x = np.meshgrid(np.arange(self.resol, dtype=int), np.arange(self.resol, dtype=int))

        # We now shift the coordinates at the position given by the pos tuple by subtracting.
        dist_x = x - self.pos[0]
        dist_y = y - self.pos[1]

        # We compute the distance between each pixel and the centre of the circle using the equation of circle.
        dist_r = np.sqrt(dist_x**2 + dist_y**2)

        dist_bool = dist_r < self.rad # Boolean values stored

        self.output[y[dist_bool],x[dist_bool]] = 1

        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        plt.show()

class Spectrum:
    def __init__(self, resol):
        """ resol (int) = Number of pixels in each dimension """
        self.resol = resol
        self.output = np.zeros((resol,resol,3),dtype=float) # 3D Array

    def draw(self):

        # Pattern arrays from 0 to 1 and 1 to 0 respectively
        forward_pat = np.linspace(0.0, 1.0, self.resol, dtype=float)
        backward_pat = np.linspace(1.0, 0.0, self.resol, dtype=float)

        # Initialize RGB array with proper shape
        rgb = np.zeros((self.resol, self.resol, 3),dtype=float)

        # Red channel: varies left to right
        rgb[:, :, 0] = np.tile(forward_pat, (self.resol, 1))

        # Green channel: varies top to bottom. Hence the transpose
        rgb[:, :, 1] = np.tile(forward_pat, (self.resol, 1)).T

        # Blue channel: varies left to right (decreasing)
        rgb[:, :, 2] = np.tile(backward_pat, (self.resol, 1))

        # rgb array copy is returned to the output array
        self.output = rgb.copy()
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        plt.show()