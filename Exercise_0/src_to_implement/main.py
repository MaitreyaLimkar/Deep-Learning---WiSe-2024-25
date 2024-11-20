"""This is the Main script that imports and calls the classes created and are implemented below."""

import pattern as pat
import generator as gen

checkerboard = pat.Checker(100, 10)
checkerboard.draw()
checkerboard.show()

circle = pat.Circle(250, 50, (90,125))
circle.draw()
circle.show()

spectrum = pat.Spectrum(200)
spectrum.draw()
spectrum.show()

generate = gen.ImageGenerator('./data/exercise_data/', './data/Labels.json',
                              10, [32, 32, 3], rotation=False, mirroring=False,
                             shuffle=True)
generate.show()