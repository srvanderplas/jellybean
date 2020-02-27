# load the requisite libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io
import numpy as np
import cmapy
import os
os.getcwd()
os.chdir("D:\Jellybean")
import functions_jellybean as fj
from skimage.segmentation import random_walker
# read in the image, convert to float
img = cv2.imread(r"D:\Jellybean\data\7UP(R).png")/255.0
img = cv2.imread(r"D:\Jellybean\data\A&W(R) Cream Soda.png")/255.0

img = cv2.imread(r"D:\Jellybean\data\Buttered Popcorn.png")/255.0


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

# convert the mask to 3d
mask_3d = np.dstack([mask]*3)


# mask on the image
img_mask = img*mask_3d


#find where the pixels are black in the mask
mask3 = (erosion == 0)

# convert the black pixels to white
img[mask3,:] =1

# see the eroded mask
cv2.imshow('mask', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# CIE 1931 luminance grayscale conversion
img = img_mask[:,:,0]*0.0722 + img_mask[:,:,1]*0.7152 + img_mask[:,:,2]*0.2126
# make 3d and then filter?
img = np.dstack([img]*3)
img[mask3,:] =1
img = img[:,:,0]


img = cv2.convertScaleAbs(img*255)

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# do histogram equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

equ = cv2.equalizeHist(img)/255.0

img = cv2.convertScaleAbs(equ*255)

# see the eroded mask
cv2.imshow('mask', cl1)
cv2.waitKey(0)
cv2.destroyAllWindows()

sure_bg, sure_fg = fj.random_walker_func(img_mask, beta=100, opening_iterations = 1)

plt.hist(img.ravel(),256,[50,256]);
plt.show()


ret2,th2 = cv2.threshold(cl1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

markers = np.zeros(img.shape, dtype=np.uint)
markers[cl1 < 0.5*ret2] = 1
markers[cl1 > ret2] = 2

# Run random walker algorithm
labels = random_walker(img, markers, beta=100, mode='bf')

img2 = color.label2rgb(labels, bg_label=0)
cv2.imshow('watershed 2d', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

image_chk = np.float64(labels - 1)
kernel = np.ones((3,3),np.uint8)


# see the eroded mask
cv2.imshow('mask', image_chk)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(image_chk,cv2.MORPH_OPEN,kernel, iterations = 10)

# see the eroded mask
cv2.imshow('mask', opening)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Finding sure foreground area
dist_transform = cv2.distanceTransform(cv2.convertScaleAbs(opening*255),cv2.DIST_L2,5)/255.0
dist_transform = cv2.convertScaleAbs(dist_transform*255)
ret2,th2 = cv2.threshold(dist_transform,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)



sure_bg = opening.copy()
sure_fg = th2.copy()
img1, markers = fj.conduct_watershed(img,sure_fg, sure_bg)
#######################################################################

# Otsu's thresholding
dist_transform = cv2.convertScaleAbs(dist_transform*255)
ret2,th2 = cv2.threshold(dist_transform,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# see the eroded mask
cv2.imshow('mask', th2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# find contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img_4 = cv2.drawContours(img, contours, -1, (0,255,0), 3)

# see the eroded mask
cv2.imshow('mask', img_4)
cv2.waitKey(0)
cv2.destroyAllWindows()

interim = fj.clean_pixel_region(img_4, first_erosion_size = 5, dilation_size = 35, second_erosion_size = 25,
                       first_erosion_iter = 1, dilation_iter = 0, second_erosion_iter = 0)

# see the eroded mask
cv2.imshow('mask', interim)
cv2.waitKey(0)
cv2.destroyAllWindows()

#areas = [cv2.contourArea(i) for i in contours]
#
#contours_reduced_idx = [i for i,j in enumerate(areas) if cv2.contourArea(contours[i]) == 0] 
#
#contours_reduced = [j for i,j in enumerate(contours) if i in contours_reduced_idx]
#reduced_img = cv2.drawContours(img, contours_reduced[1], 0, (0, 255, 0), 3) 
## see the eroded mask
#cv2.imshow('mask', reduced_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

