# we have to decide which are dark beans
# and which are light beans

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from skimage.filters import threshold_otsu
from joblib import Parallel, delayed
from demo_new_trial_light_end_to_end_catch import mask_for_beans_light_catch
from demo_new_trial_dark_end_to_end import mask_for_beans_dark

path = r"D:\Jellybean\Relevant\jellybean_data-master"
os.chdir(path)

folder_names = os.listdir()

catch = []

for i in folder_names: 
    path_temp = path + "\\" + i
    os.chdir(path_temp)
    first_file = path_temp + "\\" +  os.listdir()[0]
    
    # read the file
    test_image_colored = cv2.imread(first_file)
    # convert to yuv
    # the y channel is the light/dark channel
    test_image = cv2.cvtColor(test_image_colored, cv2.COLOR_BGR2YUV)
    # extract the y component    
    test_image = test_image[:,:,0]
    
    # get the lowest value of the pixel
    lowest_value = np.min(test_image.ravel())
    
    # catch
    catch_dict = {"name":i, "pixel":  lowest_value}
    
    # append
    catch.append(catch_dict)


# convert to a dataframe
df_val = pd.DataFrame(catch).sort_values("pixel", ascending = False).reset_index(drop= True)

# threshold_otsu(df_val["pixel"])

plt.hist(df_val["pixel"],np.max(df_val["pixel"]),[np.min(df_val["pixel"]),np.max(df_val["pixel"])])
plt.show()


# above red apple can go to light_val
# get the paths for the light_val


path = r"D:\Jellybean\Relevant\jellybean_data-master"
light_val = df_val[:30]["name"]
light_paths = []
for i in light_val: 
    path_temp = path + "\\" + i
    os.chdir(path_temp)
    first_file = path_temp + "\\" +  os.listdir()[0]
    last_file = path_temp + "\\" +  os.listdir()[len(os.listdir())-1]
    
    # append
    light_paths.append(first_file)
    light_paths.append(last_file)
    
Parallel(n_jobs=6, verbose = 10, backend = "loky")(delayed(mask_for_beans_light_catch)(i,resize_factor = 3.5) for i in light_paths) 
    
# red apple and belowcan go into the dark script
# get the paths for the dark_val


path = r"D:\Jellybean\Relevant\jellybean_data-master"
dark_val = df_val[30:]["name"]
dark_paths = []
for i in dark_val: 
    path_temp = path + "\\" + i
    os.chdir(path_temp)
    first_file = path_temp + "\\" +  os.listdir()[0]
    last_file = path_temp + "\\" +  os.listdir()[len(os.listdir())-1]
    
    # append
    dark_paths.append(first_file)
    dark_paths.append(last_file)

Parallel(n_jobs=6, verbose = 10, backend = "loky")(delayed(mask_for_beans_dark)(i,resize_factor = 0.7) for i in dark_paths)

    