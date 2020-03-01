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
import imageio
from scipy.stats import norm
from PIL import Image
from tqdm import tqdm
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
    
    
#     visualize
#    cv2.imshow('watershed 3d', img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#
#    #Let's color the labels to see the effect
#    # plot as colored on a 2d
#    img2 = color.label2rgb(markers, bg_label=0)
#    cv2.imshow('watershed 2d', img2)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    return(img, markers)
 
def random_walker_func(img, beta=100, opening_iterations = 1, cutoff_method = "otsu"):
    # CIE 1931 luminance grayscale conversion
    img = img[:,:,0]*0.0722 + img[:,:,1]*0.7152 + img[:,:,2]*0.2126

    # change to uint8? - the thresholding functions only deals with unit8
    img = cv2.convertScaleAbs(img*255)
    
    # equalize histogram
    equ = cv2.equalizeHist(img)/255.0

    img = cv2.convertScaleAbs(equ*255)
    
    if cutoff_method == "otsu":
    
        # determine thresholds, define markers
        ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
        markers = np.zeros(img.shape, dtype=np.uint)
        markers[img < ret2] = 1 
        markers[img > ret2] = 2
    elif cutoff_method == "normal": 
        mean_norm = np.mean(img.ravel())
        sd_norm = np.std(img.ravel())
        markers = np.zeros(img.shape, dtype=np.uint)
        markers[img < norm.ppf(0.05, loc=mean_norm, scale=sd_norm)] = 1 
        markers[img > norm.ppf(0.95, loc=mean_norm, scale=sd_norm)] = 2
        
        
    
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
    
def find_markers(markers, cutoff = 900):
    # define a mask same shape as the markers
    # instantiate
    catch_all = []
    
    for i in np.unique(markers):
        if i not in [1,-1]:
#            print(i)
            mask = np.zeros(markers.shape, dtype="uint8")
            mask[markers == i] = 255
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            area1 = cv2.contourArea(contours[0])
            catch = {"contour": i, "area" : area1}
            catch_all.append(catch)
    markers_ind = [i["contour"] for i in catch_all if i["area"] > cutoff]
    return(markers_ind)


# write another function to take markers image
def get_the_beans(image,contours_list, markers):
    # list
    capture = []
    for i in contours_list:
        # get the mask
        mask = np.zeros(markers.shape, dtype="uint8")
        mask[markers == i] = 255
        # fill holes in the mask
        mask = cv2.convertScaleAbs(ndimage.binary_fill_holes(mask).astype(float)*255)
        # detect contour in the mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # fit an ellipse to the contour
        e = cv2.fitEllipse(contours[0])
        # get  mask define by the ellipse
        mask=cv2.ellipse(mask, e, color=(255,255,255), thickness=-1)/255.0
        # convert the mask to 3d
        mask_3d = np.dstack([mask.astype(bool)]*3)
        segmented_bean = (image*mask_3d)
        # make background as white
#        mask_img = segmented_bean[:,:,0] != 0
#        erosion = mask_img == 0
#        segmented_bean[erosion,:] = 1
        capture.append(segmented_bean)
#        cv2.imshow('marker 1',segmented_bean)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
    return(capture)

# now can we write a function that given just the path will split the image to jellybeans
def finally_get_the_beans(path,beta=100, opening_iterations = 2,cutoff = 900,cutoff_method = "otsu"):
    path_image = path
    
    # get the file name
    file_name = path_image.split("\\")[-1].split(".")[0]
    
    # step 1
    img_mask,step0, step0_5 = fj.read_process_image(path_image)
    
    
    # watershed using random walker
    sure_bg, sure_fg = fj.random_walker_func(img_mask, beta, opening_iterations,cutoff_method = cutoff_method)
    watershed_img, markers = fj.conduct_watershed(step0_5,sure_fg, sure_bg)
    
    
    # get area using findcontours
    # remove really small areas one
    # those would be one with specs
    contours_list = fj.find_markers(markers, cutoff = 900)


    stack_images = fj.get_the_beans(step0_5,contours_list, markers)
    
    # now save it
    counter = 0
    for j in range(len(stack_images)):
        counter = counter + 1
        img = cv2.convertScaleAbs(stack_images[j].copy()*255)
        file_sub = "D:\\Jellybean\\Split_Jellybeans\\" + file_name + "_"  + str(counter) + ".png"
        cv2.imwrite(file_sub, img)


# define a function to take the paths
def get_normal_parms(path_obj):
    paths = path_obj
    # obj
    catch_here = []
    # what are the unique names
    unique_names = np.unique([i.split("\\")[-1].split("_")[-0] for i in paths])
    # then go through the image name and read in data
    for it in unique_names: 
        print(it)
        # crop and reduce image size
        stack_img = [Image.open(j).convert('RGB').crop((Image.open(j).convert('RGB').getbbox())) 
                    for j in paths if it in j]
        # convert to open cv image
        open_cv_image = [np.array(i)[:, :, ::-1] for i in stack_img]
        #convert to hsv
        hsv_image = [cv2.cvtColor(i, cv2.COLOR_BGR2HSV) for i in open_cv_image ]
        
        h_vec_app = []
        s_vec_app = []
        v_vec_app = []
        
        for i in hsv_image:
#            print(i)
            # split
            h, s, v = cv2.split(i)
        
            # make an iterable object
            zip_iter = zip(h.ravel(),s.ravel(),v.ravel())
        
            # remove black pixels
            chk = [i for i in zip_iter if np.sum(i) > 0]
        
            # get the h, s and v vectors
            h_vec = [i[0] for i in chk]
            h_vec_app.append(h_vec)
            
            s_vec = [i[1] for i in chk]
            s_vec_app.append(s_vec)
            
            v_vec = [i[2] for i in chk]
            v_vec_app.append(v_vec)
            
        
        # get the parms

        h_vec_app = [item for sub in h_vec_app for item in sub]
        s_vec_app = [item for sub in s_vec_app for item in sub]
        v_vec_app = [item for sub in v_vec_app for item in sub]
        
        h_parms = [np.mean(h_vec_app), np.std(h_vec_app)]
        s_parms = [np.mean(s_vec_app), np.std(s_vec_app)]
        v_parms = [np.mean(v_vec_app), np.std(v_vec_app)]
        
        catch = {"type": it, "h_parms": h_parms,"s_parms": s_parms, "v_parms": v_parms}
        
        catch_here.append(catch)
    return(catch_here)
    

def get_normal_parms_seg(path_obj):
    paths = path_obj
    # obj
    catch_here = []
    # what are the unique names
#    unique_names = np.unique([i.split("\\")[-1].split("_")[-0] for i in paths])
    # then go through the image name and read in data
    for it in tqdm(paths): 
#        print(it)
        # crop and reduce image size
        name = [it.split("\\")[-1].split("_")[-0]]
        stack_img = [Image.open(it).convert('RGB').crop((Image.open(it).convert('RGB').getbbox()))]
        # convert to open cv image
        open_cv_image = [cv2.cvtColor(np.array(i), cv2.COLOR_RGB2BGR) for i in stack_img]
        
        # histogram equalization of colored images
        
        hsv_image = [cv2.cvtColor(i, cv2.COLOR_BGR2HSV) for i in open_cv_image]
#        open_cv_image = [cv2.cvtColor(i, cv2.COLOR_BGR2YCrCb) for i in open_cv_image]
#        hsv_image = []
#        for img in open_cv_image: 
#            y, cr, cb = cv2.split(img)
#            # Applying equalize Hist operation on Y channel.
#            y_eq = cv2.equalizeHist(y)
#            img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
#            img_rgb_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)
#            #convert to hsv
#            hsv_image_ck = cv2.cvtColor(img_rgb_eq, cv2.COLOR_BGR2HSV)
#            hsv_image.append(hsv_image_ck)
        
        for i in hsv_image:
#            print(i)
            # split
            h, s, v = cv2.split(i)
        
            # make an iterable object
            zip_iter = zip(h.ravel(),s.ravel(),v.ravel())
        
            # remove black pixels
            chk = [i for i in zip_iter if np.sum(i) > 0]
        
            # get the h, s and v vectors
            h_vec = [i[0] for i in chk]
#            h_vec_app.append(h_vec)
            h_parms = [np.mean(h_vec), np.std(h_vec)]
            s_vec = [i[1] for i in chk]
            s_parms = [np.mean(s_vec), np.std(s_vec)]
#            s_vec_app.append(s_vec)
            v_vec = [i[2] for i in chk]
            v_parms = [np.mean(v_vec), np.std(v_vec)]
#            v_vec_app.append(v_vec)
            
            catch = {"type": name[0], "h_mean": h_parms[0], "h_std": h_parms[1], 
                     "s_mean": s_parms[0], "s_std": s_parms[1],"v_mean": v_parms[0], "v_std": v_parms[1]}
        # get the parms

        
        
        
            catch_here.append(catch)
    return(catch_here)