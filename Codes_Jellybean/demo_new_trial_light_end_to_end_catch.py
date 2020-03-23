## necessary library  imports
import cv2
# from scipy import ndimage
from skimage.segmentation import flood_fill
# from skimage.segmentation import clear_border
import numpy as np
# import matplotlib.pyplot as plt
# import os
# # os.getcwd()
# os.chdir("D:\Jellybean\Relevant\Codes")
# import functions_jellybean as fj

# from skimage import color
from skimage.filters import threshold_multiotsu
from PIL import Image
# from skimage.segmentation import random_walker
# import imutils
import uuid

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

# path = r"D:\Jellybean\Relevant\jellybean_data-master\aw_cream_soda\DSC_0002.JPG"
# path = r"D:\Jellybean\Relevant\jellybean_data-master\berry_blue\DSC_0007.JPG"
# path = r"D:\Jellybean\Relevant\jellybean_data-master\mango\DSC_0002.JPG"
# path = r"D:\Jellybean\Relevant\jellybean_data-master\sunkist_lemon\DSC_0027.JPG"
# path = r"D:\Jellybean\Relevant\jellybean_data-master\lemon_lime\DSC_0493.JPG"
# path = r"D:\Jellybean\Relevant\jellybean_data-master\red_apple\DSC_0003.JPG"

def mask_for_beans_light_catch(path, resize_factor = 3.5):

    # read the image
    test_image_colored = cv2.imread(path)
    name = path.split("\\")[-2]+"." +path.split("\\")[-1].split(".")[-2]
    # convert to yuv
    # the y channel is the light/dark channel
    test_image = cv2.cvtColor(test_image_colored, cv2.COLOR_BGR2YUV)
    # extract the y component    
    test_image = test_image[:,:,0]
    
    # equalize the histogram
    equ = cv2.equalizeHist(test_image)
    # hard to see big images in spyder
    # uncomment if you want to see
    # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\equalized_grayscale.png",equ)
    
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
    # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\mask0_gray.png",mask1)
    
    
    mask2 = np.zeros(regions.shape, dtype="uint8")
    mask2[regions == 1] = 255
    # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\mask1_gray.png",mask2)

    mask3 = np.zeros(regions.shape, dtype="uint8")
    mask3[regions == 2] = 255
    # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\mask2_gray.png",mask3)

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
        # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\thresholded_mask.png",mask)

    # flood fill at each corner to get rid of whites on edges
    
    th2 = flood_fill(mask, (0, 0),0)
    th2 = flood_fill(th2, (0, mask.shape[1]-1),0)
    th2 = flood_fill(th2, (mask.shape[0]-1, 0),0)
    th2 = flood_fill(th2, (mask.shape[0]-1, mask.shape[1]-1),0)
    # visualize
    # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\thresholded_gray_ff.png",th2)
    
    # label the contours
    ret, markers = cv2.connectedComponents(th2)
    # markers = markers+1
    # convert the equalized grayscale from earlier to float
    equ = equ/255.0
    
    # counter = 0
    for i in range(1,np.max(markers)):
                
        try:
            counter = uuid.uuid1() 
        # counter = counter + 1
        # print(i)
    # for a sample label index
            mask4 = np.zeros(markers.shape, dtype="uint8")
            mask4[markers == i] = 255
        # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\mask_samp.png",mask4)
    
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
        # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\cropped_colored.png",stack_img_col)
    
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
        # plt.hist(stack_img.ravel(),256,[1,256]); plt.show()
    
        # visualize the cropped image
        # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\cropped_eq.png",stack_img)
    
        # maybe we do not need to equalize since we had already equalized
        # but the previous equalization was on uncropped image
        # might have an impact
    
        # equalize histogram
            equ_temp = cv2.equalizeHist(stack_img)
        # visualize the equallized histogram
        # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\cropped_eq_sq.png",equ_temp)
        # histogram
        # plt.hist(equ_temp.ravel(),256,[1,256]); plt.show()
    



    # Apply multi-Otsu threshold 
    # thresholds = threshold_multiotsu(stack_img, classes=3)
            thresholds = threshold_multiotsu(equ_temp, classes=3)

    
        # regions = np.digitize(stack_img, bins=thresholds)
            regions = np.digitize(equ_temp, bins=thresholds)
    
        # get the three regions
            mask1 = np.zeros(regions.shape, dtype="uint8")
            mask1[regions == 0] = 255
            if mask1[0,0] == 255:
                kernel = np.ones((5,5),np.uint8)
                mask1 = cv2.erode(mask1,kernel,iterations = 2)
            # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\mask0_crop.png",mask1)
            else:
                kernel = np.ones((5,5),np.uint8)
                mask1 = cv2.dilate(mask1,kernel,iterations = 2)
            # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\mask0_crop.png",mask1)
    
            mask2 = np.zeros(regions.shape, dtype="uint8")
            mask2[regions == 1] = 255
            if mask2[0,0] == 255:
                kernel = np.ones((5,5),np.uint8)
                mask2 = cv2.erode(mask2,kernel,iterations = 2)
            # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\mask1_crop.png",mask2)
            else:
                kernel = np.ones((5,5),np.uint8)
                mask2 = cv2.dilate(mask2,kernel,iterations = 2)
            # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\mask1_crop.png",mask2)
    

            mask3 = np.zeros(regions.shape, dtype="uint8")
            mask3[regions == 2] = 255
            if mask3[0,0] == 255:
                kernel = np.ones((5,5),np.uint8)
                mask3 = cv2.erode(mask3,kernel,iterations = 2)
            # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\mask2_crop.png",mask3)
            else:
                kernel = np.ones((5,5),np.uint8)
                mask3 = cv2.dilate(mask3,kernel,iterations = 2)
            # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\mask2_crop.png",mask3)
    
            # combine the above masks into a list
            mask_comb = [mask1, mask2, mask3]
    
        # we will try to reduce the masks to a contour that has max area
        # initialize an object will have augmented masks
            aug_masks = []
    
        # iterate over the list
            for mask in mask_comb:

        
        
                th2 = flood_fill(mask, (0, 0),0)
                th2 = flood_fill(th2, (0, mask.shape[1]-1),0)
                th2 = flood_fill(th2, (mask.shape[0]-1, 0),0)
                th2 = flood_fill(th2, (mask.shape[0]-1, mask.shape[1]-1),0)
                mask = th2.copy()
            # detect contours
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
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
            # path = r"D:\Jellybean\Relevant\Test_Images\aug_mask_" + str(i) + ".png"
            # cv2.imwrite(path,n)
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
    
    
        # detect the contours in this mask
        # make it 3d, and get the median r,g,b of this mask
        # if any one of them is less than 100, ignore the mask
            contours_req, hierarchy_req = cv2.findContours(req_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        # areas = [cv2.contourArea(i) for i in contours_req]
    
            catch = []    
        # draw the contour with max area
            for i in range(len(contours_req)):
                mask_draw_req = np.zeros(req_mask.shape, np.uint8)
            # print(i)
                mask_draw_req = cv2.drawContours(mask_draw_req, contours_req, i, (255,255), -1)
        
                mask_3d_col_req = np.dstack([mask_draw_req]*3)
                img_mask_col_ns_req = (stack_img_col/255.0)*(mask_3d_col_req/255.0)
                img_mask_col_ns_req =  cv2.convertScaleAbs(img_mask_col_ns_req*255)
            # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\checkcheck.png",img_mask_col_ns_req)
            # red channel
                red_channel = img_mask_col_ns_req[:,:,2]
                red_channel_flat = red_channel.ravel()
                median_red_channel = [i for i in red_channel_flat if i !=0]
                median_red_channel = np.median(median_red_channel)
            # green channel
                green_channel = img_mask_col_ns_req[:,:,1]
                green_channel_flat = green_channel.ravel()
                median_green_channel = [i for i in green_channel_flat if i !=0]
                median_green_channel = np.median(median_green_channel)        
            # blue channel
                blue_channel = img_mask_col_ns_req[:,:,0]
                blue_channel_flat = blue_channel.ravel()
                median_blue_channel = [i for i in blue_channel_flat if i !=0]
                median_blue_channel = np.median(median_blue_channel)           
        
                list_catch = [median_red_channel, median_green_channel, median_blue_channel]
                how_many = [i for i in list_catch if i >= 70]
                how_many_sum = len(how_many)
                catch.append(how_many_sum)
    
            # try to get the center of the contours
        # contours_temp,_= cv2.findContours(req_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = imutils.grab_contours(contours_temp)
    
            mask_draw = np.zeros(req_mask.shape, np.uint8)
            for i,n in  enumerate(contours_req):
        
                if catch[i] > 1:
        
                    cnt_scaled = scale_contour(n, resize_factor)
                # mask_draw = np.zeros(req_mask.shape, np.uint8)
                    mask_temp = cv2.drawContours(mask_draw, [cnt_scaled], -1, (255,255), -1)
    
        # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\scaled_mask.png",mask_temp)
    

    
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
        # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\colored_no_shadow.png",img_mask_col_ns)
    
        # maybe can clean it by fitting an ellipse
        # get the mask first
            img_mask_col_ns_mask = (img_mask_col_ns[:,:,0] + img_mask_col_ns[:,:,1] + img_mask_col_ns[:,:,2]) >=1
            img_mask_col_ns_mask  = cv2.convertScaleAbs(img_mask_col_ns_mask*255)
        # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\colored_no_shadow_bin.png",img_mask_col_ns_mask.astype(float))
    
        # detect contours
            contours,hierarchy = cv2.findContours(img_mask_col_ns_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        # remove holes
            contours = [ i for i in contours if cv2.contourArea(i) > 10000]
    
        # draw contours
            mask = np.zeros(img_mask_col_ns_mask.shape, np.uint8)
            for i in contours:
                e = cv2.fitEllipse(i)
            # get  mask define by the ellipse
                mask_ellipse =cv2.ellipse(mask, e, color=(255,255,255), thickness=-1)/255.0

    
            mask_ellipse  = cv2.convertScaleAbs(mask_ellipse*255)    
        # cv2.imwrite(r"D:\Jellybean\Relevant\Test_Images\colored_elliptical_mask.png",mask_ellipse)
    
            mask_3d_col = np.dstack([mask_ellipse]*3)
    
        # multiply the 3d mask with the colored crop of the bean
        # to hopefully remove the shadow
            img_mask_col_ns = (stack_img_col/255.0)*(mask_3d_col/255.0)
            img_mask_col_ns = cv2.convertScaleAbs(img_mask_col_ns*255)
    
            file_sub = "D:\\Jellybean\\Relevant\\Test_Images\\split\\" + name + "_"  + str(counter) + ".png"
        # visualize
            cv2.imwrite(file_sub   ,img_mask_col_ns)
    
        except:
            pass
# mask_for_beans(path)

