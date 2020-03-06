# set directory to D:Jellybean
import os
os.getcwd()
os.chdir("D:\Jellybean")
import functions_jellybean as fj
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import glob
from PIL import Image
import pandas as pd
from  tqdm import tqdm
from collections import Counter
from joblib import Parallel, delayed
from joblib import parallel_backend


# read and process
#step0, step0_5 = fj.read_process_image(r"D:\Jellybean\data\A&W(R) Cream Soda.png")

#step0, step0_5 = fj.read_process_image(r"D:\Jellybean\data\7UP(R).png")
#path_image = r"D:\Jellybean\data\Cappuccino.png"
#path_image = r"D:\Jellybean\data\A&W(R) Cream Soda.png"
#path_image = r"D:\Jellybean\data\7UP(R).png"
#path_image = r"D:\Jellybean\data\Smoothie Blend.png"
#path_image = r"D:\Jellybean\data\Mango.png"
#path_image = r"D:\Jellybean\data\Birthday Cake Remix(TM).png"
#path_image = r"D:\Jellybean\data\Sunkist(R) Orange.png"
#path_image = r"D:\Jellybean\data\Strawberry Cheesecake.png"
#path_image = r"D:\Jellybean\data\Very Cherry.png"
#path_image = r"D:\Jellybean\data\Red Apple.png"
#fj.finally_get_the_beans(path_image, beta = 10, opening_iterations = 2, cutoff_method = "otsu")

# readall the paths
# readall the paths
paths = glob.glob("D:\Jellybean\data\*.png")

counter = 0
for i in paths: 
    fj.finally_get_the_beans(i, beta = 10, opening_iterations = 2, cutoff_method = "otsu")
    counter = counter + 1
    print(counter)

# now we will have to read the split images    
# read the data in, get images of the same beans - then fit
# a normal curve and then plot for (h, s and v values)
paths = glob.glob("D:\Jellybean\Split_Jellybeans\*.png")

# segmented beans
#result = fj.get_normal_parms_seg(paths)
#with parallel_backend('multiprocessing', n_jobs=2):
#    result = Parallel(verbose = 10)(delayed(fj.get_normal_parms_seg_pr)(i, gamma = 2) for i in paths)
result = Parallel(n_jobs=6, verbose = 10, backend = "loky")(delayed(fj.get_normal_parms_seg_pr)(i, gamma = 2) for i in paths)
df_beans = pd.DataFrame(result)
df_beans.to_csv("segmented_beans_parms.csv", index = False)


# segmented beans
#result = fj.get_normal_parms_seg_rgb(paths)
result = Parallel(n_jobs=6, verbose = 10, backend = "loky")(delayed(fj.get_normal_parms_seg_rgb_pr)(i, gamma = 2) for i in paths)
df_beans = pd.DataFrame(result)
df_beans.to_csv("segmented_beans_parms_rgb.csv", index = False)

# segmented beans
#result = fj.get_normal_parms_seg_yuv(paths)
result = Parallel(n_jobs=6, verbose = 10, backend = "loky")(delayed(fj.get_normal_parms_seg_yuv_pr)(i, gamma = 2) for i in paths)
df_beans = pd.DataFrame(result)
df_beans.to_csv("segmented_beans_parms_yuv.csv", index = False)

# k = 3
# color features - kmeans - rgb
result = Parallel(n_jobs=6, verbose = 10, backend = "loky")(delayed(fj.k_means_cluster_RGB)(i, n_clusters = 3, gamma = 2)
         for i in paths)
df_beans = pd.DataFrame(result)
df_beans.to_csv("kmeans_rgb_features.csv", index = False)

# color features - kmeans - hsv
result = Parallel(n_jobs=6, verbose = 10, backend = "loky")(delayed(fj.k_means_cluster_HSV)(i, type_img = "HSV",n_clusters = 3,
                  gamma = 2)for i in paths)
df_beans = pd.DataFrame(result)
df_beans.to_csv("kmeans_hsv_features.csv", index = False)

#gamma = 1
#result = fj.get_normal_parms_seg(paths)
#with parallel_backend('multiprocessing', n_jobs=2):
#    result = Parallel(verbose = 10)(delayed(fj.get_normal_parms_seg_pr)(i, gamma = 2) for i in paths)
result = Parallel(n_jobs=6, verbose = 10, backend = "loky")(delayed(fj.get_normal_parms_seg_pr)(i, gamma = 1) for i in paths)
df_beans = pd.DataFrame(result)
df_beans.to_csv("segmented_beans_parms_1.csv", index = False)


# segmented beans
#result = fj.get_normal_parms_seg_rgb(paths)
result = Parallel(n_jobs=6, verbose = 10, backend = "loky")(delayed(fj.get_normal_parms_seg_rgb_pr)(i, gamma = 1) for i in paths)
df_beans = pd.DataFrame(result)
df_beans.to_csv("segmented_beans_parms_rgb_1.csv", index = False)

# segmented beans
#result = fj.get_normal_parms_seg_yuv(paths)
result = Parallel(n_jobs=6, verbose = 10, backend = "loky")(delayed(fj.get_normal_parms_seg_yuv_pr)(i, gamma = 1) for i in paths)
df_beans = pd.DataFrame(result)
df_beans.to_csv("segmented_beans_parms_yuv_1.csv", index = False)

# k = 3
# color features - kmeans - rgb
result = Parallel(n_jobs=6, verbose = 10, backend = "loky")(delayed(fj.k_means_cluster_RGB)(i, n_clusters = 3, gamma = 1)
         for i in paths)
df_beans = pd.DataFrame(result)
df_beans.to_csv("kmeans_rgb_features_1.csv", index = False)

# color features - kmeans - hsv
result = Parallel(n_jobs=6, verbose = 10, backend = "loky")(delayed(fj.k_means_cluster_HSV)(i, type_img = "HSV",n_clusters = 3,
                  gamma = 1)for i in paths)
df_beans = pd.DataFrame(result)
df_beans.to_csv("kmeans_hsv_features_1.csv", index = False)



## k = 2
## color features - kmeans - rgb
#result = Parallel(n_jobs=6, verbose = 10, backend = "loky")(delayed(fj.k_means_cluster)(i, n_clusters = 2) for i in paths)
#df_beans = pd.DataFrame(result)
#df_beans.to_csv("kmeans_rgb_features.csv", index = False)
#
## color features - kmeans - hsv
#result = Parallel(n_jobs=6, verbose = 10, backend = "loky")(delayed(fj.k_means_cluster)(i, type_img = "HSV",n_clusters = 2)
#                     for i in paths)
#df_beans = pd.DataFrame(result)
#df_beans.to_csv("kmeans_hsv_features.csv", index = False)


## k = 3
## color features - kmeans - rgb
#result = Parallel(n_jobs=6, verbose = 10)(delayed(fj.k_means_cluster,  backend = "loky")(i, n_clusters = 3) for i in paths)
#df_beans = pd.DataFrame(result)
#df_beans.to_csv("kmeans_rgb_features.csv", index = False)
#
## color features - kmeans - hsv
#result = Parallel(n_jobs=6, verbose = 10)(delayed(fj.k_means_cluster,  backend = "loky")(i, type_img = "HSV",n_clusters = 3)
#                     for i in paths)
#df_beans = pd.DataFrame(result)
#df_beans.to_csv("kmeans_hsv_features.csv", index = False)
#



# categories
result = fj.get_normal_parms(paths)
df_beans = pd.DataFrame(result)
df_beans.to_csv("beans_parms.csv")

######################### methods ############################################
# first method
img_mask,step0, step0_5 = fj.read_process_image(path_image)
# watershed using random walker
sure_bg, sure_fg = fj.random_walker_func(img_mask, beta=10, opening_iterations = 2)
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


stack_images = fj.get_the_beans(step0_5,contours_list, markers)
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
#cv2.imshow('marker 1',hsv_image[0])
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
#cv2.imshow('marker 1',np.bitwise_or(img, mask_3d))
#cv2.waitKey(0)
#cv2.destroyAllWindows()

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