from skimage.io import imread
import matplotlib.pyplot as plt
import libWordDetection as wd
a = wd.pyWordDetection()

img = imread('001.tif')
c_range = range(1, 4)
r_range = range(1, 3)

a.extract_regions(img, c_range, r_range)
