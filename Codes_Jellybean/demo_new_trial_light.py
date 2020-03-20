## necessary library  imports
import cv2
from scipy import ndimage
from skimage.segmentation import flood_fill
# from skimage.segmentation import clear_border
import numpy as np
import matplotlib.pyplot as plt
import os
os.getcwd()
os.chdir("D:\Jellybean\Relevant\Codes")
import functions_jellybean as fj

from skimage import color
from skimage.filters import threshold_multiotsu
from PIL import Image
from skimage.segmentation import random_walker
import imutils

# functions from github 
def max_white(nimg):
    if nimg.dtype==np.uint8:
        brightest=float(2**8)
    elif nimg.dtype==np.uint16:
        brightest=float(2**16)
    elif nimg.dtype==np.uint32:
        brightest=float(2**32)
    else:
        brightest==float(2**8)
    nimg = nimg.transpose(2, 0, 1)
    nimg = nimg.astype(np.int32)
    nimg[0] = np.minimum(nimg[0] * (brightest/float(nimg[0].max())),255)
    nimg[1] = np.minimum(nimg[1] * (brightest/float(nimg[1].max())),255)
    nimg[2] = np.minimum(nimg[2] * (brightest/float(nimg[2].max())),255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def retinex_adjust(nimg):
    """
    from 'Combining Gray World and Retinex Theory for Automatic White Balance in Digital Photography'
    """
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    sum_r = np.sum(nimg[0])
    sum_r2 = np.sum(nimg[0]**2)
    max_r = nimg[0].max()
    max_r2 = max_r**2
    sum_g = np.sum(nimg[1])
    max_g = nimg[1].max()
    coefficient = np.linalg.solve(np.array([[sum_r2,sum_r],[max_r2,max_r]]),
                                  np.array([sum_g,max_g]))
    nimg[0] = np.minimum((nimg[0]**2)*coefficient[0] + nimg[0]*coefficient[1],255)
    sum_b = np.sum(nimg[1])
    sum_b2 = np.sum(nimg[1]**2)
    max_b = nimg[1].max()
    max_b2 = max_r**2
    coefficient = np.linalg.solve(np.array([[sum_b2,sum_b],[max_b2,max_b]]),
                                             np.array([sum_g,max_g]))
    nimg[1] = np.minimum((nimg[1]**2)*coefficient[0] + nimg[1]*coefficient[1],255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

# increase, decrease the size of a contour
def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


def mask_for_beans(path):

    # try with one image first
    # sample paths
    # dark and light images
    # path = r"D:\Jellybean\Relevant\jellybean_data-master\wild_blackberry\DSC_0493.JPG"
    path = r"D:\Jellybean\Relevant\jellybean_data-master\aw_cream_soda\DSC_0002.JPG"

    # read the image
    test_image_colored = cv2.imread(path)
    
    # convert to yuv
    # the y channel is the light/dark channel
    test_image = cv2.cvtColor(test_image_colored, cv2.COLOR_BGR2YUV)
    # extract the y component    
    test_image = test_image[:,:,0]
    
    # equalize the histogram
    equ = cv2.equalizeHist(test_image)
    # hard to see big images in spyder
    # uncomment if you want to see
    cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\equalized_grayscale.png",equ)
    
    # Apply multi-Otsu threshold
    # there are three basic components
    # the bean, the shadow, and the background
    thresholds = threshold_multiotsu(equ, classes=3)
    
    # Using the threshold values, we generate the three regions.
    regions = np.digitize(equ, bins=thresholds)
    
    # basic strategy
    # initialize an empty mask
    # and use the regions to get different masks
    mask1 = np.zeros(regions.shape, dtype="uint8")
    mask1[regions == 0] = 255
    # visualize
    cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\mask0_gray.png",mask1)
    
    
    mask2 = np.zeros(regions.shape, dtype="uint8")
    mask2[regions == 1] = 255
    cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\mask1_gray.png",mask2)

    mask3 = np.zeros(regions.shape, dtype="uint8")
    mask3[regions == 2] = 255
    cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\mask2_gray.png",mask3)

    # choose the mask which has lowest variation in the areas of detected contours
    # after let's say truncating some low area contours
    
    # our main goal is to automatically pick the correct mask
    # we want 
    # 1. the index for the contours that have area more than 100000
    # 2. the areas of those contours
    
    # my hypothesis is that, the best mask would have
    # the lowest variation in the areas of the contours
    
    # empty list for the index of each contour in each mask    
    # empty list for the index of each contour in each mask    
    obj_catch1 = []
    
    # contour object for each mask
    contours_catch = []
    
    # lengths for the contours
    len_catch = []
    
    # iterate through the three masks
    for i in range(3):
        # extracting mask depending on the index
        mask = (i ==0)*mask1 + (i==1)*mask2 + (i==2)*mask3
        
        # copy mask object
        sure_bg =mask.copy().astype(float)
        
        # define kernel for erosion
        kernel = np.ones((5,5),np.uint8)
        # erode
        erosion = cv2.erode(sure_bg,kernel,iterations = 1)
        # convert to uint8
        opening = cv2.convertScaleAbs(erosion*255)


        # mask = np.zeros(opening.shape, np.uint8)
        # find contours
        contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # append for each mask
        contours_catch.append(contours)
        
        # areas of contours
        areas = [cv2.contourArea(n) for i,n in enumerate(contours) if cv2.contourArea(n) > 100000]
        
        # length
        length = len(areas)
        
        # index of contours
        obj = [i for i,n in enumerate(contours) if cv2.contourArea(n) > 100000]
        
        # append the indexes
        obj_catch1.append(obj)
        
        # append the lengths
        len_catch.append(length)



    # make the masks into a list
    masks = [mask1, mask2, mask3] 
    
    # make a dictionary 
    # of indexes and areas of contours corresponding
    # to those indexes
       
    # dictionary = dict(zip([0,1,2], areas_catch))
    
    # # find the best index
    # # best index is the index of the mask
    # # that has the lowest variation in the areas of the contours
    # best_ind = [k for (k,v) in dictionary.items() if v == np.min([i for i in dictionary.values()])][0]
    best_ind = np.argmax(len_catch)
    
    # extract the mask, the index, contour object for the best index
    required_mask = masks[best_ind]
    required_obj = obj_catch1[best_ind]
    required_contours = contours_catch[best_ind]
    
    # draw the contours of the mask 
    # initialize empty mask
    mask = np.zeros(required_mask.shape, np.uint8)
    
    # iterate over the indexes
    for i in required_obj:
        # draw the picked up contours
        cv2.drawContours(mask, required_contours, i, (255,255), -1) 
        # visualize
        cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\thresholded_mask.png",mask)

    # flood fill at each corner to get rid of whites on edges
    
    th2 = flood_fill(mask, (0, 0),0)
    th2 = flood_fill(th2, (0, mask.shape[1]-1),0)
    th2 = flood_fill(th2, (mask.shape[0]-1, 0),0)
    th2 = flood_fill(th2, (mask.shape[0]-1, mask.shape[1]-1),0)
    # visualize
    cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\thresholded_gray_ff.png",th2)

    # contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # obj_areas = [cv2.contourArea(n) for i,n in enumerate(contours)]
    # obj = [i for i,n in enumerate(contours)]
    # obj = [i for i,n in enumerate(contours) if cv2.contourArea(n) > 5000]
    # mask = np.zeros(opening.shape, np.uint8)
    # for i in obj:
    #     print(i)
    #     cv2.drawContours(mask, contours, i, (255,255), -1) 
    #     #th2 = ndimage.binary_fill_holes(th2)     
    # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\thresholded.png",mask)
    
    # label the contours
    ret, markers = cv2.connectedComponents(th2)
    # markers = markers+1
    
    # for a sample label index
    mask4 = np.zeros(markers.shape, dtype="uint8")
    # mask4[markers == 5] = 255
    # mask4[markers == 6] = 255
    mask4[markers == 7] = 255
    cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\mask_samp.png",mask4)
    
    # crop the colored portion of the image as well
    mask_3d = np.dstack([mask4]*3) 
    img_mask_col = (test_image_colored/255.0)*(mask_3d/255.0)
    img_mask_col = cv2.convertScaleAbs(img_mask_col*255)
    # img_1 = Image.fromarray(np.uint8(cm.gist_earth(img_mask)))
    
    # crop the image
    # colored
    im_col = Image.fromarray(img_mask_col)
    im_box_col = im_col.getbbox()
    # translation from Image to numpy
    stack_img_col =np.array(im_col.crop(im_box_col))
    cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\cropped_colored.png",stack_img_col)
# convert the mask to 3d
    # mask_3d = np.dstack([mask4]*3)

# img = test_image[:,:,0]*0.0722 + test_image[:,:,1]*0.7152 + test_image[:,:,2]*0.2126
# img = cv2.convertScaleAbs(img*255)

# mask on the image
    
    # path = r"D:\Jellybean\Relevant\jellybean_data-master\wild_blackberry\DSC_0493.JPG"
    # path = r"D:\Jellybean\Relevant\jellybean_data-master\aw_cream_soda\DSC_0002.JPG"

    # # read the image
    # test_image = cv2.imread(path)
    # test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2YUV)
    # # test_image = retinex_adjust(test_image)
    # # test_image = max_white(test_image)
    
    # test_image = test_image[:,:,0]
    # # equ = cv2.equalizeHist(test_image)/255.0
    # # equ = cv2.convertScaleAbs(equ*255)
    # # test_image = cv2.convertScaleAbs(test_image*255)
    # # test_image = cv2.bilateralFilter(test_image, d = 10, sigmaColor = 100, sigmaSpace = 10)
    # # th2 = flood_fill(test_image, (0, 0),0)
    # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\og.png",equ)
    # y = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
    # y = cv2.convertScaleAbs(y*255)
    
    # convert the equalized grayscale from earlier to float
    equ = equ/255.0
    # equ = cv2.convertScaleAbs(equ*255)
    
    # multiply with the sample mask to get the relevant image
    img_mask = equ*(mask4/255.0)
    img_mask = cv2.convertScaleAbs(img_mask*255)
    # img_1 = Image.fromarray(np.uint8(cm.gist_earth(img_mask)))
    
    # crop the image
    im = Image.fromarray(img_mask)
    im_box = im.getbbox()
    # translation from Image to numpy
    stack_img =np.array(im.crop(im_box))
    # histogram
    plt.hist(stack_img.ravel(),256,[1,256]); plt.show()
    
    # visualize the cropped image
    cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\cropped_eq.png",stack_img)
    
    # maybe we do not need to equalize since we had already equalized
    # but the previous equalization was on uncropped image
    # might have an impact
    
    # equalize histogram
    equ = cv2.equalizeHist(stack_img)
    # visualize the equallized histogram
    cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\cropped_eq_sq.png",equ)
    # histogram
    plt.hist(equ.ravel(),256,[1,256]); plt.show()
    
#     markers = np.zeros(equ.shape, dtype=np.uint8)
#     markers[stack_img < 23] = 1
#     markers[(equ >= 30)] = 2
#     # markers[(equ > 150) & (equ < 200)] = 3

#     # Run random walker algorithm
#     labels = random_walker(stack_img, markers, beta=1, mode = "bf")
    
#     # markers = np.zeros(equ.shape, dtype=np.uint8)
#     # markers[labels == 2] = 255
#     # opening = cv2.morphologyEx(markers,cv2.MORPH_OPEN,kernel, 2)
#     img2 = color.label2rgb(labels)
#     img2 = cv2.convertScaleAbs(img2*255)
    


# # plt.hist(dst.ravel(),256,[200,256]); plt.show()
# # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\thresholded.png",dst)
#     cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\cc.png",img2)



    # Apply multi-Otsu threshold 
    # thresholds = threshold_multiotsu(stack_img, classes=3)
    thresholds = threshold_multiotsu(equ, classes=3)
    # thresholds = threshold_otsu(img_mask)

    # img_mask = img_mask >= thresholds
    # img_mask = cv2.convertScaleAbs(img_mask*255)
    # img_mask = ndimage.binary_fill_holes(img_mask)
    # img_mask = cv2.convertScaleAbs(img_mask*255)
    
    # dist_transform = cv2.distanceTransform(cv2.convertScaleAbs(img_mask),cv2.DIST_L2,5)/255.0
    # dist_transform = cv2.convertScaleAbs(dist_transform*255)
    # ret2,th2 = cv2.threshold(dist_transform,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\img.png",stack_img)
    
    # Using the threshold values, we generate the three regions.
    # have to think of how to extract the best mask from these
    # with the goal of getting a mask that is atleast approximately
    # getting the shadowy area
    
    # regions = np.digitize(stack_img, bins=thresholds)
    regions = np.digitize(equ, bins=thresholds)
    
    # get the three regions
    mask1 = np.zeros(regions.shape, dtype="uint8")
    mask1[regions == 0] = 255
    if mask1[0,0] == 255:
        kernel = np.ones((5,5),np.uint8)
        mask1 = cv2.erode(mask1,kernel,iterations = 2)
    else:
        kernel = np.ones((5,5),np.uint8)
        mask1 = cv2.dilate(mask1,kernel,iterations = 2)
    cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\mask0_crop.png",mask1)
    
    mask2 = np.zeros(regions.shape, dtype="uint8")
    mask2[regions == 1] = 255
    if mask2[0,0] == 255:
        kernel = np.ones((5,5),np.uint8)
        mask2 = cv2.erode(mask2,kernel,iterations = 2)
    else:
        kernel = np.ones((5,5),np.uint8)
        mask2 = cv2.dilate(mask2,kernel,iterations = 2)
    cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\mask1_crop.png",mask2)
    

    mask3 = np.zeros(regions.shape, dtype="uint8")
    mask3[regions == 2] = 255
    if mask3[0,0] == 255:
        kernel = np.ones((5,5),np.uint8)
        mask3 = cv2.erode(mask3,kernel,iterations = 2)
    else:
        kernel = np.ones((5,5),np.uint8)
        mask3 = cv2.dilate(mask3,kernel,iterations = 2)
    cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\mask2_crop.png",mask3)
    
    # combine the above masks into a list
    mask_comb = [mask1, mask2, mask3]
    
    # we will try to reduce the masks to a contour that has max area
    # initialize an object will have augmented masks
    aug_masks = []
    
    # iterate over the list
    for mask in mask_comb:
        # flood fill
        
        # if (0,0) pixel are white - then erode to thicken the boundary 
        # if (0,0) pixel are black - then dilate to thicken the boundary
        # if mask[0,0] == 0:
        #     kernel = np.ones((5,5),np.uint8)
        #     mask = cv2.erode(mask,kernel,iterations = 2)
        # else:
        #    kernel = np.ones((5,5),np.uint8)
        #    mask = cv2.dilate(mask,kernel,iterations = 2)
        
        
        
        th2 = flood_fill(mask, (0, 0),0)
        th2 = flood_fill(th2, (0, mask.shape[1]-1),0)
        th2 = flood_fill(th2, (mask.shape[0]-1, 0),0)
        th2 = flood_fill(th2, (mask.shape[0]-1, mask.shape[1]-1),0)
        mask = th2.copy()
        # detect contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # get areas of each contour
        # obj_areas = [cv2.contourArea(n) for i,n in enumerate(contours)]
        # get the index of the contour that has the maximum area
        # obj = [i for i,n in enumerate(contours) if cv2.contourArea(n) == np.max((obj_areas))]
        
        # this help to deal with overlapping beans
        obj = [i for i,n in enumerate(contours) if cv2.contourArea(n) >= 1000]
        
        # if the length of index is greater than 2
        # that is more than 2 contours detected, keep only two (possibly)
        
        # initialize empty mask
        mask_draw = np.zeros(mask.shape, np.uint8)
        
        # draw the contour with max area
        for i in obj:
        # print(i)
            mask_temp = cv2.drawContours(mask_draw, contours, i, (255,255), -1)
        aug_masks.append(mask_temp)
            
        #th2 = ndimage.binary_fill_holes(th2)     
    # iterate over the augmented masks and draw them on disk
    # maybe also get the area of each augmented mask
    area_aug_mask = []
    for i,n in enumerate(aug_masks):
        path = r"D:\Jellybean\Relevant\Test_Images\aug_mask_" + str(i) + ".png"
        cv2.imwrite(path,n)
        # detect contour
        contours, hierarchy = cv2.findContours(n, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        
        # area of the contour
        area = [cv2.contourArea(i) for i in contours]
        area_aug_mask.append(area)
    
    # index for the mask corresponding to smallest area
    # smallest_index   = np.argmin(area_aug_mask)
    smallest_index   = 0
    
    # get the mask corresponding to the index
    req_mask = aug_masks[smallest_index]
    
    
    # try to get the center of the contours
    contours_temp,_= cv2.findContours(req_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(contours_temp)
    
    mask_draw = np.zeros(req_mask.shape, np.uint8)
    for i in  contours_temp:   
        cnt_scaled = scale_contour(i, 1.9)
        # mask_draw = np.zeros(req_mask.shape, np.uint8)
        mask_temp = cv2.drawContours(mask_draw, [cnt_scaled], -1, (255,255), -1)
    
    cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\scaled_mask.png",mask_temp)
    
    # dilate the mask a little
    # kernel = np.ones((5,5),np.uint8)
    # dilation = cv2.dilate(req_mask,kernel,iterations = 22)
    
    # # visualize the dilated mask
    # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\dilated_mask.png",dilation)
    
    # fit an ellipse to deal with curvy and weird edges
    # contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # fit an ellipse to the contour
    # e = cv2.fitEllipse(contours[0])
    # # get  mask define by the ellipse
    # mask = np.zeros(dilation.shape, dtype="uint8")
    # mask=cv2.ellipse(mask, e, color=(255,255,255), thickness=-1)/255.0
    # mask  = cv2.convertScaleAbs(mask*255)
    
    # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\ellipse_mask.png",mask)
    
    # invert the mask
    # because we want to remove this area
    dilation_inv = 255 - mask_temp
    
    # make this a 3d mask
    # because we will apply this on the colored crop of the bean
    mask_3d_col = np.dstack([dilation_inv]*3)
    
    # multiply the 3d mask with the colored crop of the bean
    # to hopefully remove the shadow
    img_mask_col_ns = (stack_img_col/255.0)*(mask_3d_col/255.0)
    img_mask_col_ns = cv2.convertScaleAbs(img_mask_col_ns*255)
    
    # visualize
    cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\colored_no_shadow.png",img_mask_col_ns)
    
    # maybe can clean it by fitting an ellipse
    
    # import imutils
    # # multiply with the image
    # check = dilation_inv*equ
    # contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnt = contours[0]
    # epsilon = 0.001*cv2.arcLength(cnt,True)
    # approx = cv2.approxPolyDP(cnt,epsilon,True)
    # mask = np.zeros(mask.shape, np.uint8)
    # approx = cv2.drawContours(mask, [approx], -1, (255, 0, 0), 2) 
    # contours, hierarchy = cv2.findContours(approx, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # mask = np.zeros(mask.shape, np.uint8)
    # approx = cv2.fillPoly(mask, pts =[contours[0]], color=(255,255,255))
    # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\dilate.png",approx)
    # edges = cv2.Canny(check,100,200)
    # erosion = cv2.erode(check,kernel,iterations = 1)
    # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\dilate.png",approx)
    # thresholds = threshold_multiotsu(erosion, classes=2)
    # regions = np.digitize(erosion, bins=thresholds)
    # img_mask = cv2.convertScaleAbs(regions*255)
    # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\dilate.png",img_mask)
    # contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # mask = np.zeros(mask.shape, np.uint8)
    # hull= cv2.convexHull(contours[0])
    # check_hull = cv2.drawContours(mask, [hull], -1, (255, 0, 0), 2)
    # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\dilate.png",check_hull)
    
    
    
    
#     mask = np.zeros(regions.shape, dtype="uint8")
#     mask[regions == 1] = 255
#     cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\mask1.png",mask)

#     mask = np.zeros(regions.shape, dtype="uint8")
#     mask[regions == 2] = 255
#     cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\mask2.png",mask)
    
    
#     # will canny help
    
    
# #erosion = clear_border(erosion)
# sure_bg = mask.copy()



# # erosion = cv2.erode(mask,kernel,iterations = 1)  
# # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\thresholded.png",erosion)

# # cv2.imshow('watershed 2d', mask)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()



# # img2 = color.label2rgb(regions)
# # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\multi_thresh.png",img2)

# # cv2.imshow('watershed 2d', img2)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# # dst =  cv2.medianBlur(img,51)

# # markers = np.zeros(dst.shape, dtype=np.uint8)
# # markers[dst < 110] = 1
# # markers[dst > 240] = 2
# # markers[(dst > 200) & (dst < 230)] = 3

# # # Run random walker algorithm
# # labels = random_walker(dst, markers, beta=100, mode = "cg_mg")
# # from skimage import measure, color, io
# #     #Let's color the labels to see the effect
# #     # plot as colored on a 2d
# # img2 = color.label2rgb(labels, bg_label=0)
# # cv2.imshow('watershed 2d', img2)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
    


# # plt.hist(dst.ravel(),256,[200,256]); plt.show()
# # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\thresholded.png",dst)
# # try contour detection
# # mask = np.zeros(img.shape, np.uint8)
# #     # find contours
# # contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# # areas = [cv2.contourArea(n) for i,n in enumerate(contours)]
# # plt.hist(areas); plt.show()
# # obj = [i for i,n in enumerate(contours) if cv2.contourArea(n) > 1000]

# # for i in obj:
# #     cv2.drawContours(mask, contours, i, (255,255), -1) 
# # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\thresholded.png",th2)


# # ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# #erosion = clear_border(th2)
# #th2 = ndimage.binary_fill_holes(th2)
# #th2 = flood_fill(th2.astype(float), (0, 0),0)

# sure_bg =th2.copy().astype(float)
# kernel = np.ones((5,5),np.uint8)
# #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# #res = cv2.morphologyEx(sure_bg,cv2.MORPH_OPEN,kernel)
# erosion = cv2.erode(sure_bg,kernel,iterations = 1)
# opening = cv2.convertScaleAbs(erosion*255)

# cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\thresholded.png",th2)

# #opening = cv2.morphologyEx(opening,cv2.MORPH_OPEN,kernel, 2)
# th2 = flood_fill(opening, (0, 0),255)

# th2 = ndimage.binary_fill_holes(opening)
# dist_transform = cv2.convertScaleAbs(th2*255)


# dist_transform = cv2.distanceTransform(cv2.convertScaleAbs(dist_transform*255),cv2.DIST_L2,5)/255.0
# dist_transform = cv2.convertScaleAbs(dist_transform*255)
# ret2,th2 = cv2.threshold(dist_transform,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\thresholded.png",th2)


# #opening = dist_transform.copy()
# cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\thresholded.png",dist_transform)
# mask = np.zeros(opening.shape, np.uint8)
#     # find contours
# contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# obj = [i for i,n in enumerate(contours) if cv2.contourArea(n) > 200]

# for i in obj:
#     cv2.drawContours(mask, contours, i, (255,255), -1) 
# cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\thresholded.png",mask)
# kernel = np.ones((5,5),np.uint8)
# #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# #res = cv2.morphologyEx(sure_bg,cv2.MORPH_OPEN,kernel)
# erosion = cv2.erode(mask,kernel,iterations = 1)  
# cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\thresholded.png",erosion)
# th2 = flood_fill(erosion, (0, 0),0)
# th2 = flood_fill(th2, (0, 5999),0)
# th2 = flood_fill(th2, (3999, 0),0)
# th2 = flood_fill(th2, (3999, 5999),0)
# cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\thresholded.png",th2)
# contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# obj = [i for i,n in enumerate(contours) if cv2.contourArea(n) > 5000]
# mask = np.zeros(opening.shape, np.uint8)
# for i in obj:
#     cv2.drawContours(mask, contours, i, (255,255), -1) 
# #th2 = ndimage.binary_fill_holes(th2)     
# cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\thresholded.png",mask)
# #erosion = clear_border(erosion)
# sure_bg = mask.copy()

# # get the sure_fg
# dist_transform = cv2.distanceTransform(cv2.convertScaleAbs(sure_bg*255),cv2.DIST_L2,5)/255.0
# dist_transform = cv2.convertScaleAbs(dist_transform*255)
# ret2,th2 = cv2.threshold(dist_transform,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\thresholded.png",th2)

# dist_transform = cv2.distanceTransform(cv2.convertScaleAbs(th2*255),cv2.DIST_L2,5)/255.0
# dist_transform = cv2.convertScaleAbs(dist_transform*255)
# ret2,th2 = cv2.threshold(dist_transform,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# dist_transform = cv2.distanceTransform(cv2.convertScaleAbs(th2*255),cv2.DIST_L2,5)/255.0
# dist_transform = cv2.convertScaleAbs(dist_transform*255)
# ret2,th2 = cv2.threshold(dist_transform,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# sure_fg = th2.copy()
# cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\thresholded.png",sure_fg)
# file_name = "wild_berry"
# img, markers = fj.conduct_watershed(test_image, sure_fg, sure_bg)
# contours_list = fj.find_markers(markers, cutoff = 100)
# capture = fj.get_the_beans(test_image,contours_list, markers)
# mask = np.zeros(markers.shape, dtype="uint8")
# capture = []
# image = test_image.copy()
# for i in contours_list:
#     # get the mask
#     mask = np.zeros(markers.shape, dtype="uint8")
#     mask[markers == i] = 255
#     mask_3d = np.dstack([mask.astype(bool)]*3)
#     segmented_bean = (image*mask_3d)
#     capture.append(segmented_bean)
# #    cv2.imshow('marker 1',segmented_bean)
# #    cv2.waitKey(0)
# #    cv2.destroyAllWindows()
#         # fill holes in the mask
# #    mask = cv2.convertScaleAbs(ndimage.binary_fill_holes(mask).astype(float)*255)


# contours, hierarchy = cv2.findContours(markers, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# obj = [i for i,n in enumerate(contours) if cv2.contourArea(n) > 80]

# for i in obj:
#     cv2.drawContours(mask, contours, i, (255,255), -1) 




# capture = fj.get_the_beans(test_image,contours_list, markers)
# counter = 0
# for j in range(len(capture)):
#     counter = counter + 1
#     img = cv2.convertScaleAbs(capture[j].copy()*255)
# #    img = capture[j].copy()
#     file_sub = r"D:\\Jellybean\\Relevant\\Test_Images\\split\\" + file_name + "_"  + str(counter) + ".png"
#     cv2.imwrite(file_sub, img)


# # erode and fill?
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# erosion = cv2.erode(th2,kernel,iterations = 3)
# erosion = cv2.erode(sure_bg,kernel,iterations = 1)


# cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\thresholded.png",mask)


