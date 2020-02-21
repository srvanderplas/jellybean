# load the requisite libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io
import numpy as np
import cmapy

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


# trying to replicate after imlst_remask statement

# make the brush first
kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(251,251))

# normalise the brush
kern = kern/np.sum(kern)

# apply the kernel above in a convolution 
dst = cv2.filter2D(img,-1,kern)

# convert the blurry white to proper white
dst[mask3,:] =1

# visualize
cv2.imshow('mask', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find all points where the image is more intense than the background plus a fudge factor 
imlst_adapt = (img > dst + 0.075)

#combine the results of each color channel to get a single
# region of intense color
imlst_adapt_mask = (imlst_adapt[:,:,0] + imlst_adapt[:,:,1] + imlst_adapt[:,:,2]) >=1

# visualize
cv2.imshow('mask', imlst_adapt_mask.astype(float))
cv2.waitKey(0)
cv2.destroyAllWindows()

# replicate the cleanimage function 
# first fill the holes
fill_holes = ndimage.binary_fill_holes(imlst_adapt_mask)

# visualize
cv2.imshow('mask', fill_holes.astype(float))
cv2.waitKey(0)
cv2.destroyAllWindows()


# then erode
kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
erosion = cv2.erode(fill_holes.astype(float),kern,iterations = 1)

# visualize
cv2.imshow('mask', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

# then dilate
kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(35,35))
dilate = cv2.dilate(erosion,kern,iterations = 1)

# visualize
cv2.imshow('mask', dilate)
cv2.waitKey(0)
cv2.destroyAllWindows()

# fill holes again
fill_holes = ndimage.binary_fill_holes(dilate)

# visualize
cv2.imshow('mask', fill_holes.astype(float))
cv2.waitKey(0)
cv2.destroyAllWindows()

# erode again
kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
erosion = cv2.erode(fill_holes.astype(float),kern,iterations = 1)

# visualize
cv2.imshow('mask', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

# get the regions defined by the mask
masked_data = cv2.bitwise_and(cv2.convertScaleAbs(img*255)
, cv2.convertScaleAbs(img*255)
, mask=cv2.convertScaleAbs(erosion*255)
)

# visualize
cv2.imshow('mask', masked_data)
cv2.waitKey(0)
cv2.destroyAllWindows()

# highlight mask on image like R code
# convert the erosion mask to 3d
erosion1 = np.dstack([erosion]*3)
overlay = cv2.addWeighted(cv2.convertScaleAbs(img*255),0.5,cv2.convertScaleAbs(erosion1*255),0.5,0)

# visualize
cv2.imshow('mask', overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

########### code replication from R complete for one jellybean########
########## have to apply to all jellybeans#####################


########## watershed attempt #################
# Finding sure foreground area

# sure background area
# then dilate
kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(35,35))
sure_bg = cv2.dilate(erosion,kern,iterations = 5)
# visualize
cv2.imshow('mask', sure_bg)
cv2.waitKey(0)
cv2.destroyAllWindows()


# then erode
kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
erosion = cv2.erode(erosion,kern,iterations = 6)




# visualize
cv2.imshow('mask', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

# then dilate
kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(35,35))
dilate = cv2.dilate(erosion,kern,iterations = 1)

# visualize
cv2.imshow('mask', dilate)
cv2.waitKey(0)
cv2.destroyAllWindows()



dist_transform = cv2.distanceTransform(cv2.convertScaleAbs(dilate*255),cv2.DIST_L2,5)/255
# visualize
cv2.imshow('mask', dist_transform)
cv2.waitKey(0)
cv2.destroyAllWindows()

ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),1,0)

sure_fg = dilate.copy()
# visualize
cv2.imshow('mask', sure_fg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Finding unknown region
# instead of this as the foreground
# let's actually use the dilated image above as the sure foreground
sure_fg = np.uint8(sure_fg)
sure_fg = dilate.copy()
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(cv2.convertScaleAbs(sure_bg*255),sure_fg)

# visualize
cv2.imshow('mask', unknown)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

img = cv2.convertScaleAbs(img*255)


markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]


# visualize
cv2.imshow('mask', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Let's color the labels to see the effect
# plot as colored on a 2d
img2 = color.label2rgb(markers, bg_label=0)
cv2.imshow('mask', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


########## from here on is the code to try a floodfill #####################

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
cv2.imshow('mask', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

