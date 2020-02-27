# convert the last code to functions
# import the relevant libraries
# load the requisite libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io
import numpy as np
import cmapy
import functions_jellybean as fj
from skimage.segmentation import random_walker
# define a function to read in the image and apply the masks
def read_process_image(path): 
    # read in the image, convert to float
    img = cv2.imread(path)/255.0
    # read in the mask, convert to float
    mask = cv2.imread(r"D:\Jellybean\inst\JellyBellyMask.png",0)/255.0
    
    # convert the mask to 3d
    mask_3d = np.dstack([mask]*3)
    
    # mask on the image
    img_mask = img*mask_3d
    
    # erode the mask as in the R code?
    # get the strucuting element?

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31))
    erosion = cv2.erode(mask,kern,iterations = 1)
    
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
    
    # Find all points where the image is more intense than the background plus a fudge factor 
    imlst_adapt = (img > dst + 0.075)
    
    #combine the results of each color channel to get a single
    # region of intense color
    imlst_adapt_mask = (imlst_adapt[:,:,0] + imlst_adapt[:,:,1] + imlst_adapt[:,:,2]) >=1
    
    # return
    return(img_mask,imlst_adapt_mask, img)

def clean_pixel_region(image, first_erosion_size = 5, dilation_size = 35, second_erosion_size = 25,
                       first_erosion_iter = 1, dilation_iter = 1, second_erosion_iter = 1): 
    # first fill the holes
    fill_holes = ndimage.binary_fill_holes(image)
    
    # then erode
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(first_erosion_size,first_erosion_size))
    erosion = cv2.erode(fill_holes.astype(float),kern,iterations = first_erosion_iter)
    # then dilate
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilation_size,dilation_size))
    dilate = cv2.dilate(erosion,kern,iterations = dilation_iter)
    # fill holes again
    fill_holes = ndimage.binary_fill_holes(dilate)
    # erode again
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(second_erosion_size,second_erosion_size))
    erosion = cv2.erode(fill_holes.astype(float),kern,iterations = second_erosion_iter)
    
    return(erosion)

def visualize_centers_on_image(path, centers): 
    img = cv2.imread(path)/255.0
    # get the regions defined by the mask
    masked_data = cv2.bitwise_and(cv2.convertScaleAbs(img*255)
    , cv2.convertScaleAbs(img*255)
    , mask=cv2.convertScaleAbs(centers*255))
    
    # highlight mask on image like R code
    # convert the erosion mask to 3d
    erosion1 = np.dstack([centers]*3)
    overlay = cv2.addWeighted(cv2.convertScaleAbs(img*255),0.7,cv2.convertScaleAbs(erosion1*255),0.3,0)
    
    # visualize
    cv2.imshow('centers 2d', masked_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # visualize
    cv2.imshow('centers 3d', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_background(img,image, dilation_size = 35, dilation_iter = 0):
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilation_size,dilation_size))
    sure_bg = cv2.dilate(image,kern,iterations = dilation_iter)
    # visualize
    cv2.imshow('sure background', sure_bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # highlight mask on image like R code
    # convert the erosion mask to 3d
    sure_bg1 = np.dstack([sure_bg]*3)
    overlay = cv2.addWeighted(cv2.convertScaleAbs(img*255),0.7,cv2.convertScaleAbs(sure_bg1*255),0.3,0)
    
    # visualize
    # visualize
    cv2.imshow('background overlayed', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return(sure_bg)

def get_foreground(img,image, erosion_size = 5, erosion_iter = 1, dilation_size = 35, dilation_iter = 1):
    # then erode
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(erosion_size,erosion_size))
    erosion = cv2.erode(image,kern,iterations = erosion_iter)
    
    # then dilate
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilation_size,dilation_size))
    dilate = cv2.dilate(erosion,kern,iterations = dilation_iter)


    # visualize
    cv2.imshow('sure foreground', dilate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # convert the erosion mask to 3d
    sure_fg = np.dstack([dilate]*3)
    overlay = cv2.addWeighted(cv2.convertScaleAbs(img*255),0.7,cv2.convertScaleAbs(sure_fg*255),0.3,0)
    
    # visualize
    # visualize
    cv2.imshow('foreground overlayed', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return(dilate)    

def get_foreground_dist_transform(img,image, distance_type = cv2.DIST_L2, mask_size = 5, cutoff = 0.7, dilation_size= 35, dilation_iter=1, erosion_size = 5, erosion_iter = 1):
    # then erode
    dist_transform = cv2.distanceTransform(cv2.convertScaleAbs(image*255),cv2.DIST_L2,5)/255
    ret, sure_fg = cv2.threshold(dist_transform,cutoff*dist_transform.max(),1,0)
    
    # then erode
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(erosion_size,erosion_size))
    erosion = cv2.erode(sure_fg,kern,iterations = erosion_iter)
    
    # then dilate
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilation_size,dilation_size))
    dilate = cv2.dilate(erosion,kern,iterations = dilation_iter)
    
    # sure_fg 
    sure_fg = dilate
    # visualize
    cv2.imshow('sure foreground dist transform', sure_fg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # convert the erosion mask to 3d
    sure_fg1 = np.dstack([sure_fg]*3)
    overlay = cv2.addWeighted(cv2.convertScaleAbs(img*255),0.6,cv2.convertScaleAbs(sure_fg1*255),0.4,0)
    
    # visualize
    # visualize
    cv2.imshow('foreground overlayed', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return(np.float64(sure_fg))  

def conduct_watershed(img,sure_fg, sure_bg):
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(cv2.convertScaleAbs(sure_bg*255),sure_fg)
    
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
    cv2.imshow('watershed 3d', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Let's color the labels to see the effect
    # plot as colored on a 2d
    img2 = color.label2rgb(markers, bg_label=0)
    cv2.imshow('watershed 2d', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return(img, markers)
 
def random_walker_func(img, beta=100, opening_iterations = 1):
    # CIE 1931 luminance grayscale conversion
    img = img[:,:,0]*0.0722 + img[:,:,1]*0.7152 + img[:,:,2]*0.2126

    # change to uint8? - the thresholding functions only deals with unit8
    img = cv2.convertScaleAbs(img*255)
    
    # equalize histogram
    equ = cv2.equalizeHist(img)/255.0

    img = cv2.convertScaleAbs(equ*255)
    
    # determine thresholds, define markers
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    markers = np.zeros(img.shape, dtype=np.uint)
    markers[img < ret2] = 1 
    markers[img > ret2] = 2
    
    # Run random walker algorithm
    labels = random_walker(img, markers, beta, mode='bf')
    
    # prepare background
    # convert to grayscale
    image_chk = np.float64(labels - 1)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(image_chk,cv2.MORPH_OPEN,kernel, opening_iterations)
    
    # sure foreground
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(cv2.convertScaleAbs(opening*255),cv2.DIST_L2,5)/255.0
    dist_transform = cv2.convertScaleAbs(dist_transform*255)
    ret2,th2 = cv2.threshold(dist_transform,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    sure_bg = opening.copy()
    sure_fg = th2.copy()
    
    # maybe erode and open for foreground
#    kernel = np.ones((3,3),np.uint8)
#    opening = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, kernel)
    
    
    
#    img1, markers = fj.conduct_watershed(img,sure_fg, sure_bg)
    
    return(sure_bg, sure_fg)
    
        