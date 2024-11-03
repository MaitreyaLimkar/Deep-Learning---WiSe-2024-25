"""This is the Main script that imports and calls the classes created and are implemented below.
    As per instructions, no loops are used."""

import pattern as pat
from Exercise_0.src_to_implement.pattern import Spectrum

checkerboard = pat.Checker(100, 10)
checkerboard.draw()
checkerboard.show()

circle = pat.Circle(250, 50, (90,125))
circle.draw()
circle.show()

spectrum = pat.Spectrum(200)
spectrum.draw()
spectrum.show()