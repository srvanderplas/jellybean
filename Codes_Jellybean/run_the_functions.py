# set directory to D:Jellybean
import os
os.getcwd()
os.chdir("D:\Jellybean")
import functions_jellybean as fj
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

# read and process
#step0, step0_5 = fj.read_process_image(r"D:\Jellybean\data\A&W(R) Cream Soda.png")

#step0, step0_5 = fj.read_process_image(r"D:\Jellybean\data\7UP(R).png")
path_image = r"D:\Jellybean\data\Cappuccino.png"
path_image = r"D:\Jellybean\data\A&W(R) Cream Soda.png"
path_image = r"D:\Jellybean\data\7UP(R).png"
path_image = r"D:\Jellybean\data\Smoothie Blend.png"
path_image = r"D:\Jellybean\data\Mango.png"
path_image = r"D:\Jellybean\data\Birthday Cake Remix(TM).png"
img_mask,step0, step0_5 = fj.read_process_image(path_image)

# first method

# watershed using random walker
sure_bg, sure_fg = fj.random_walker_func(img_mask, beta=100, opening_iterations = 1)
watershed_img, markers = fj.conduct_watershed(step0_5,sure_fg, sure_bg)
# how many markers
np.unique(markers)

# how to get image for one particular marker
# see the eroded mask
#foreground mask

# get area using findcontours
# remove really small areas one
# those would be one with specs
contours_list = fj.find_markers(markers, cutoff = 900)


fj.get_the_beans(step0_5,contours_list, markers)
# first method ends


# second method
# clean pixel region
step1 = fj.clean_pixel_region(step0,first_erosion_size = 5)

# visualize
fj.visualize_centers_on_image(path_image,step1)

# background
step2 = fj.get_background(step0_5,image = step1, dilation_size = 30, dilation_iter = 2)

# foreground using centers/dist transform approach
step3 = fj.get_foreground(step0_5,step1, erosion_size = 10, erosion_iter = 2, dilation_size = 40, dilation_iter = 2)

#step3 = fj.get_foreground_dist_transform(step0_5,step1,distance_type = cv2.DIST_L2, mask_size = 5, cutoff = 0.2, dilation_size= 35, dilation_iter=1, erosion_size = 5, erosion_iter = 1)

# watershed
watershed_img, markers = fj.conduct_watershed(step0_5,step3, step2)

# second method ends

#    
#mask=cv2.ellipse(mask, e, color=(255,255,255), thickness=-1)/255.0
## make 3d
## convert the mask to 3d
#mask_3d = np.dstack([mask.astype(bool)]*3)
#
#segmented_bean = step0_5*mask_3d
#
#cv2.imshow('marker 1',segmented_bean)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#mask = np.zeros(markers.shape, dtype="uint8")
#mask[markers == 30] = 255
#cv2.imshow('marker 1', ndimage.binary_fill_holes(mask).astype(float))
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.contourArea(contours[0])
##cv2.drawContours(step0_5, contours, -1, (255, 255, 255), 3)
#e = cv2.fitEllipse(contours[0])
#cv2.ellipse(step0_5, e, (255,0, 255), 1, cv2.LINE_AA)
#cv2.imshow('marker 1',contours[0])
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
## let's try to get one particular bean
#bean_4_mask = (markers == 4)
#
## convert the mask to 3d
#mask_3d = np.dstack([bean_4_mask]*3)
#
## bean
#bean_4 = mask_3d*step0_5
#
### extract a particular marker
#cv2.imshow('marker 1', bean_4)
#cv2.waitKey(0)
#cv2.destroyAllWindows()



# colored histogram
color = ('b','g','r')
for channel,col in enumerate(color):
    histr = cv2.calcHist([cv2.convertScaleAbs(bean_4*255)],[channel],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([1,256])
    plt.ylim([0,2000])
plt.title('Histogram for color scale picture')
plt.show()

