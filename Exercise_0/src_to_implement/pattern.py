"""This file is used to implement the classes Checker and Circle resolpectively.
    We will implement them using the constructor method __init__(), a method draw() to create the pattern,
    and finally a visualization function show()."""

import numpy as np
import matplotlib.pyplot as plt

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
        self.output = None

    def draw(self):

    def show(self):
