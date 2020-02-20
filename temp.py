# load the requisite libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io
import numpy as np

# read in the image, convert to float
img = cv2.imread(r"D:\Jellybean\data\7UP(R).png")/255.0

# read in the mask, convert to float
mask = cv2.imread(r"D:\Jellybean\inst\JellyBellyMask.png",0)/255.0

# see the mask? 
cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# erode the mask as in the R code?
# get the strucuting element?

kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31))
erosion = cv2.erode(mask,kern,iterations = 1)

# see the eroded mask
cv2.imshow('mask', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

#find where the pixels are black in the mask
mask3 = (erosion == 0)

# convert the black pixels to white
img[mask3,:] =1

# CIE 1931 luminance grayscale conversion
img = img[:,:,0]*0.0722 + img[:,:,1]*0.7152 + img[:,:,2]*0.2126

# change to uint8? - the thresholding functions only deals with unit8
img = cv2.convertScaleAbs(img*255)

# adaptive thresholding
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,61,0.00001*255)

# invert the image
th2 = 255 - th2

# convert back to float
th2 = th2/255.0

# multiply the image by the eroded mask
img = th2*erosion

# Copy the thresholded image.
im_floodfill = cv2.convertScaleAbs(img.copy()*255)
 
# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = img.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0,0), 255);

# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# Combine the two images to get the foreground.
im_out = cv2.convertScaleAbs(img.copy()*255) | im_floodfill_inv

# do the connected components that gives colors to the beans, 
# seems to capture three distinct beans

s = [[1,1,1],[1,1,1],[1,1,1]]
#label_im, nb_labels = ndimage.label(mask)
labeled_mask, num_labels = ndimage.label(im_out, structure=s)

#The function outputs a new image that contains a different integer label 
#for each object, and also the number of objects found.


#Let's color the labels to see the effect
img2 = color.label2rgb(labeled_mask, bg_label=0)
cv2.imshow('mask', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()