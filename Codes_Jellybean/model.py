# Number of features to consider at every split
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from skimage import measure, color, io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import xgboost as xgb
import sklearn.discriminant_analysis as da
from collections import Counter 
from sklearn.neighbors import NearestCentroid
from skimage.segmentation import flood, flood_fill
# set directory to D:Jellybean
import os
import pickle
os.getcwd()
os.chdir("D:\Jellybean")
import functions_jellybean as fj
#from joblib import Parallel, delayed
# hsv features
seg_beans = pd.read_csv("D:\Jellybean\segmented_beans_parms.csv")

# make additional features
seg_beans["h_ratio"] = seg_beans["h_mean"]/seg_beans["h_std"]
seg_beans["s_ratio"] = seg_beans["s_mean"]/seg_beans["s_std"]
seg_beans["v_ratio"] = seg_beans["v_mean"]/seg_beans["v_std"]

seg_beans["hs_ratio"] = seg_beans["h_mean"]/seg_beans["s_mean"]
seg_beans["hv_ratio"] = seg_beans["h_mean"]/seg_beans["v_mean"]
seg_beans["sv_ratio"] = seg_beans["s_mean"]/seg_beans["v_mean"]

seg_beans["hsd_ratio"] = seg_beans["h_std"]/seg_beans["s_std"]
seg_beans["hvd_ratio"] = seg_beans["h_std"]/seg_beans["v_std"]
seg_beans["svd_ratio"] = seg_beans["s_std"]/seg_beans["v_std"]

# yuv features
seg_beans_yuv = pd.read_csv("D:\Jellybean\segmented_beans_parms_yuv.csv")

# make additional features
seg_beans_yuv["y_ratio"] = seg_beans_yuv["y_mean"]/seg_beans_yuv["y_std"]
seg_beans_yuv["u_ratio"] = seg_beans_yuv["u_mean"]/seg_beans_yuv["u_std"]
seg_beans_yuv["v_ratio"] = seg_beans_yuv["v_mean"]/seg_beans_yuv["v_std"]

seg_beans_yuv["yu_ratio"] = seg_beans_yuv["y_mean"]/seg_beans_yuv["u_mean"]
seg_beans_yuv["yv_ratio"] = seg_beans_yuv["y_mean"]/seg_beans_yuv["v_mean"]
seg_beans_yuv["uv_ratio"] = seg_beans_yuv["u_mean"]/seg_beans_yuv["v_mean"]

seg_beans_yuv["yusd_ratio"] = seg_beans_yuv["y_std"]/seg_beans_yuv["u_std"]
seg_beans_yuv["yvsd_ratio"] = seg_beans_yuv["y_std"]/seg_beans_yuv["v_std"]
seg_beans_yuv["uvsd_ratio"] = seg_beans_yuv["u_std"]/seg_beans_yuv["v_std"]

# rgb features
seg_beans_rgb = pd.read_csv("D:\Jellybean\segmented_beans_parms_rgb.csv")

# make additional features
seg_beans_rgb["b_ratio"] = seg_beans_rgb["b_mean"]/seg_beans_rgb["b_std"]
seg_beans_rgb["g_ratio"] = seg_beans_rgb["g_mean"]/seg_beans_rgb["g_std"]
seg_beans_rgb["r_ratio"] = seg_beans_rgb["r_mean"]/seg_beans_rgb["r_std"]

seg_beans_rgb["bg_ratio"] = seg_beans_rgb["b_mean"]/seg_beans_rgb["g_mean"]
seg_beans_rgb["br_ratio"] = seg_beans_rgb["b_mean"]/seg_beans_rgb["r_mean"]
seg_beans_rgb["gr_ratio"] = seg_beans_rgb["g_mean"]/seg_beans_rgb["r_mean"]

seg_beans_rgb["bgsd_ratio"] = seg_beans_rgb["b_std"]/seg_beans_rgb["g_std"]
seg_beans_rgb["brsd_ratio"] = seg_beans_rgb["b_std"]/seg_beans_rgb["r_std"]
seg_beans_rgb["grsd_ratio"] = seg_beans_rgb["g_std"]/seg_beans_rgb["r_std"]


# kmeans rgb
seg_beans_rgb_kmeans = pd.read_csv("D:\Jellybean\kmeans_rgb_features.csv")


# kmeans hsv
seg_beans_hsv_kmeans = pd.read_csv("D:\Jellybean\kmeans_hsv_features.csv")


combined = pd.concat([seg_beans,seg_beans_rgb.iloc[:,1:], seg_beans_yuv.iloc[:,1:], seg_beans_rgb_kmeans, 
                      seg_beans_hsv_kmeans], axis =1)


# gamma = 1
seg_beans = pd.read_csv("D:\Jellybean\segmented_beans_parms_1.csv")

# make additional features
seg_beans["h_ratio"] = seg_beans["h_mean"]/seg_beans["h_std"]
seg_beans["s_ratio"] = seg_beans["s_mean"]/seg_beans["s_std"]
seg_beans["v_ratio"] = seg_beans["v_mean"]/seg_beans["v_std"]

seg_beans["hs_ratio"] = seg_beans["h_mean"]/seg_beans["s_mean"]
seg_beans["hv_ratio"] = seg_beans["h_mean"]/seg_beans["v_mean"]
seg_beans["sv_ratio"] = seg_beans["s_mean"]/seg_beans["v_mean"]

seg_beans["hsd_ratio"] = seg_beans["h_std"]/seg_beans["s_std"]
seg_beans["hvd_ratio"] = seg_beans["h_std"]/seg_beans["v_std"]
seg_beans["svd_ratio"] = seg_beans["s_std"]/seg_beans["v_std"]

# yuv features
seg_beans_yuv = pd.read_csv("D:\Jellybean\segmented_beans_parms_yuv_1.csv")

# make additional features
seg_beans_yuv["y_ratio"] = seg_beans_yuv["y_mean"]/seg_beans_yuv["y_std"]
seg_beans_yuv["u_ratio"] = seg_beans_yuv["u_mean"]/seg_beans_yuv["u_std"]
seg_beans_yuv["v_ratio"] = seg_beans_yuv["v_mean"]/seg_beans_yuv["v_std"]

seg_beans_yuv["yu_ratio"] = seg_beans_yuv["y_mean"]/seg_beans_yuv["u_mean"]
seg_beans_yuv["yv_ratio"] = seg_beans_yuv["y_mean"]/seg_beans_yuv["v_mean"]
seg_beans_yuv["uv_ratio"] = seg_beans_yuv["u_mean"]/seg_beans_yuv["v_mean"]

seg_beans_yuv["yusd_ratio"] = seg_beans_yuv["y_std"]/seg_beans_yuv["u_std"]
seg_beans_yuv["yvsd_ratio"] = seg_beans_yuv["y_std"]/seg_beans_yuv["v_std"]
seg_beans_yuv["uvsd_ratio"] = seg_beans_yuv["u_std"]/seg_beans_yuv["v_std"]

# rgb features
seg_beans_rgb = pd.read_csv("D:\Jellybean\segmented_beans_parms_rgb_1.csv")

# make additional features
seg_beans_rgb["b_ratio"] = seg_beans_rgb["b_mean"]/seg_beans_rgb["b_std"]
seg_beans_rgb["g_ratio"] = seg_beans_rgb["g_mean"]/seg_beans_rgb["g_std"]
seg_beans_rgb["r_ratio"] = seg_beans_rgb["r_mean"]/seg_beans_rgb["r_std"]

seg_beans_rgb["bg_ratio"] = seg_beans_rgb["b_mean"]/seg_beans_rgb["g_mean"]
seg_beans_rgb["br_ratio"] = seg_beans_rgb["b_mean"]/seg_beans_rgb["r_mean"]
seg_beans_rgb["gr_ratio"] = seg_beans_rgb["g_mean"]/seg_beans_rgb["r_mean"]

seg_beans_rgb["bgsd_ratio"] = seg_beans_rgb["b_std"]/seg_beans_rgb["g_std"]
seg_beans_rgb["brsd_ratio"] = seg_beans_rgb["b_std"]/seg_beans_rgb["r_std"]
seg_beans_rgb["grsd_ratio"] = seg_beans_rgb["g_std"]/seg_beans_rgb["r_std"]


# kmeans rgb
seg_beans_rgb_kmeans = pd.read_csv("D:\Jellybean\kmeans_rgb_features_1.csv")


# kmeans hsv
seg_beans_hsv_kmeans = pd.read_csv("D:\Jellybean\kmeans_hsv_features_1.csv")


combined_1 = pd.concat([seg_beans,seg_beans_rgb.iloc[:,1:], seg_beans_yuv.iloc[:,1:], seg_beans_rgb_kmeans, 
                      seg_beans_hsv_kmeans], axis =1)


#combined = pd.concat([combined, combined_1])
#combined = seg_beans
combined = seg_beans_rgb
#combined_1 = pd.concat([seg_beans,seg_beans_rgb.iloc[:,1:]], axis =1)
#combined = pd.concat([combined_1])


le = preprocessing.LabelEncoder()

train_features = combined.iloc[:,1:]
train_labels = combined[combined.columns[0]]
le.fit(train_labels)
train_labels = le.transform(train_labels) 
le.inverse_transform([1])
pickle.dump(le, open("le.sav", 'wb'))
# random forest fittting and tuning

n_estimators = [int(x) for x in np.arange(1,1002,100)]
#max_depth = np.arange(1, 16,5)
#max_features = ["auto"]
#min_samples_leaf = np.arange(1,5)/10
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier(random_state = 42)
# Create the random grid
random_grid = {'n_estimators': n_estimators}
#               'max_features': max_features,
#               'max_depth': max_depth}
#               'min_samples_split': min_samples_split,
#               'min_samples_leaf': min_samples_leaf,
#               'bootstrap': bootstrap}

rf_random = GridSearchCV(estimator = rf, param_grid = random_grid, cv = 5, verbose=10, n_jobs = 6)
# Fit the random search model
rf_random.fit(train_features, train_labels)
rf_random.best_estimator_
rf_random.best_score_
rf_random.best_params_




pickle.dump(rf_random, open("random_forest.sav", 'wb'))

result_imp = np.transpose(pd.DataFrame([train_features.columns, rf_random.best_estimator_.feature_importances_]))

result_imp = result_imp.sort_values(by=[1], ascending = False)





# Create the random grid
lr = rf = RandomForestClassifier()
max_depth = np.arange(1, 32)
random_grid = {'n_estimators': [141], 
#               'max_features': max_features,
               'max_depth': max_depth}
#               'min_samples_split': min_samples_split,
#               'min_samples_leaf': min_samples_leaf,
#               'bootstrap': bootstrap}

rf_random = GridSearchCV(estimator = rf, param_grid = random_grid, cv = 5, verbose=10, n_jobs = 6)
# Fit the random search model
rf_random.fit(train_features, train_labels)
rf_random.best_estimator_
rf_random.best_score_
rf_random.best_params_

catch = []
for feature in zip(train_features.columns, rf_random.best_estimator_.feature_importances_):
    catch.append(feature)
#    print(feature)

df = pd.DataFrame(catch).sort_values(by = 1, ascending = False)

# elastic net 
# Get column names first
names = train_features.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(train_features)
scaled_df = pd.DataFrame(scaled_df, columns=names)

lr = LogisticRegression(penalty = "elasticnet", multi_class = "multinomial", solver = "saga")

random_grid = {'C': np.arange(0.01,101,10), 
               "l1_ratio": np.arange(1,101,9)/100}
#               'max_features': max_features,
#               'max_depth': max_depth,
#               'min_samples_split': min_samples_split,
#               'min_samples_leaf': min_samples_leaf,
#               'bootstrap': bootstrap}
lr_random = GridSearchCV(estimator = lr, param_grid = random_grid, cv = 5, verbose=10, n_jobs = 6)
lr_random.fit(scaled_df, train_labels)
lr_random.best_estimator_
lr_random.best_score_
lr_random.best_params_

# knearest neighbors
knn = KNeighborsClassifier()

random_grid = {'n_neighbors': np.arange(1,11,1), 
               "weights": ["uniform", "distance"]}
#               'max_features': max_features,
#               'max_depth': max_depth,
#               'min_samples_split': min_samples_split,
#               'min_samples_leaf': min_samples_leaf,

knn_random = GridSearchCV(estimator = knn, param_grid = random_grid, cv = 5, verbose=10, n_jobs = 6)

knn_random.fit(scaled_df, train_labels)
knn_random.best_estimator_
knn_random.best_score_
knn_random.best_params_


# qda
qda = da.QuadraticDiscriminantAnalysis(priors = 
                                       [Counter(train_labels)[i]/len(np.unique(train_labels)) for i in Counter(train_labels)])

random_grid = {'reg_param': np.arange(1,11,1)/10}
#               'max_features': max_features,
#               'max_depth': max_depth,
#               'min_samples_split': min_samples_split,
#               'min_samples_leaf': min_samples_leaf,

qda_random = GridSearchCV(estimator = qda, param_grid = random_grid, cv = 5, verbose=10, n_jobs = 6)

qda_random.fit(scaled_df, train_labels)
qda_random.best_estimator_
qda_random.best_score_
qda_random.best_params_

#nearest centroid
nc = NearestCentroid()
random_grid = {'metric':["euclidean", "manhattan"]}
nc_random = GridSearchCV(estimator = nc, param_grid = random_grid, cv = 5, verbose=10, n_jobs = 6)

nc_random.fit(scaled_df, train_labels)
nc_random.best_estimator_
nc_random.best_score_
nc_random.best_params_



# test
# try classifying an image

# we have to convert background of jellybean image to black

# kmeans to get the dominant colors
# and there proportions
# read the split jellybean

# write a function to get the cluster centers 


    
obj = k_means_cluster(path, type_img = "HSV")




# have to choose k
# choose k using gap statistic
#def optimalK(data, nrefs=3, maxClusters=15):
#    """
#    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
#    Params:
#        data: ndarry of shape (n_samples, n_features)
#        nrefs: number of sample reference datasets to create
#        maxClusters: Maximum number of clusters to test for
#    Returns: (gaps, optimalK)
#    """
#    gaps = np.zeros((len(range(1, maxClusters)),))
#    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
#    for gap_index, k in enumerate(range(1, maxClusters)):
#
#        # Holder for reference dispersion results
#        refDisps = np.zeros(nrefs)
#
#        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
#        for i in range(nrefs):
#            
#            # Create new random reference set
#            randomReference = np.random.random_sample(size=data.shape)
#            
#            # Fit to it
#            km = KMeans(k)
#            km.fit(randomReference)
#            
#            refDisp = km.inertia_
#            refDisps[i] = refDisp
#
#        # Fit cluster to original data and create dispersion
#        km = KMeans(k)
#        km.fit(data)
#        
#        origDisp = km.inertia_
#
#        # Calculate gap statistic
#        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
#
#        # Assign this loop's gap statistic to gaps
#        gaps[gap_index] = gap
#        
#        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)
#
#    return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal
#
#k, gapdf = optimalK(data_kmeans, nrefs=5, maxClusters=15)





path = r"D:\Jellybean\Test_Images\Tutti-Fruitti.png"
path = r"D:\Jellybean\Test_Images\Strawberry-Jam-Signature.png"
path = r"D:\Jellybean\Test_Images\Red-Apple-Signature-ns_fg.png"
path = r"D:\Jellybean\Test_Images\Watermelon-Signature-ns_fg.png"
path = r"D:\Jellybean\Test_Images\aw_CREAM_sODA.PNG"
path = r"D:\Jellybean\Test_Images\ff.png"
test_image = Image.open(path).convert('RGB').crop((Image.open(path).convert('RGB').getbbox())) 
open_cv_image = np.array(test_image)[:, :, ::-1]
#cv2.floodFill(test_image, None, seedPoint=(0,0), newVal=(0, 0, 0))
rf_random.best_estimator_


# Floodfill
#seed_point = (0, 0)
#cv2.floodFill(open_cv_image, None, seedPoint=seed_point, newVal=(36, 255, 12), loDiff=(0, 0, 0, 0), upDiff=(0, 0, 0, 0))
#
#
# reading in single image and making background black
test_image = cv2.imread(r"D:\Jellybean\Test_Images\pomegranate-jelly-belly-beans.png")
path = r"D:\Jellybean\Test_Images\pomegranate-jelly-belly-beans.png"
test_image = cv2.imread(path,0)
ret2,th2 = cv2.threshold(test_image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
from scipy import ndimage
th2 = cv2.convertScaleAbs(ndimage.binary_fill_holes(th2).astype(float)*255)
mask_3d = np.dstack([th2.astype(bool)]*3)
test_image = cv2.imread(r"D:\Jellybean\Test_Images\pomegranate-jelly-belly-beans.png")
new_img = test_image*mask_3d
open_cv_image = new_img

#cv2.putText(open_cv_image,"Hello World!!!", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

#open_cv_image = cv2.cvtColor(np.array(new_img), cv2.COLOR_RGB2BGR)
#open_cv_image = open_cv_image[:, :, ::-1]
#open_cv_image = adjust_gamma(open_cv_image, gamma) 
#    open_cv_image = open_cv_image[:, :, ::-1]



h, s, v = cv2.split(open_cv_image)

name = ["pomogranate"]
        
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
#           s_vec_app.append(s_vec)
v_vec = [i[2] for i in chk]
v_parms = [np.mean(v_vec), np.std(v_vec)]
#            v_vec_app.append(v_vec)
            
catch = { "b_mean": h_parms[0], "b_std": h_parms[1], 
                     "g_mean": s_parms[0], "g_std": s_parms[1],"r_mean": v_parms[0], "r_std": v_parms[1]}


test_df = pd.DataFrame(catch, index = [0])

seg_beans_rgb = test_df

seg_beans_rgb["b_ratio"] = seg_beans_rgb["b_mean"]/seg_beans_rgb["b_std"]
seg_beans_rgb["g_ratio"] = seg_beans_rgb["g_mean"]/seg_beans_rgb["g_std"]
seg_beans_rgb["r_ratio"] = seg_beans_rgb["r_mean"]/seg_beans_rgb["r_std"]

seg_beans_rgb["bg_ratio"] = seg_beans_rgb["b_mean"]/seg_beans_rgb["g_mean"]
seg_beans_rgb["br_ratio"] = seg_beans_rgb["b_mean"]/seg_beans_rgb["r_mean"]
seg_beans_rgb["gr_ratio"] = seg_beans_rgb["g_mean"]/seg_beans_rgb["r_mean"]

seg_beans_rgb["bgsd_ratio"] = seg_beans_rgb["b_std"]/seg_beans_rgb["g_std"]
seg_beans_rgb["brsd_ratio"] = seg_beans_rgb["b_std"]/seg_beans_rgb["r_std"]
seg_beans_rgb["grsd_ratio"] = seg_beans_rgb["g_std"]/seg_beans_rgb["r_std"]


le.inverse_transform([rf_random.best_estimator_.predict(seg_beans_rgb)])

# trying watershed


# get background first
    # read the chunk
path = r"D:\Jellybean\Test_Images\trynna_1.PNG"
test_image = cv2.imread(path)/255.0

# grayscale and threshold
img = test_image[:,:,0]*0.0722 + test_image[:,:,1]*0.7152 + test_image[:,:,2]*0.2126
dist_transform = cv2.convertScaleAbs(img*255)
equ = cv2.equalizeHist(dist_transform)/255.0
img = cv2.convertScaleAbs(equ*255)
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


th2 = ndimage.binary_fill_holes(th2)
sure_bg =th2.copy().astype(float)

# now get sure_fg
img = test_image[:,:,0]*0.0722 + test_image[:,:,1]*0.7152 + test_image[:,:,2]*0.2126
dist_transform = cv2.convertScaleAbs(img*255)
equ = cv2.equalizeHist(dist_transform)/255.0
img = cv2.convertScaleAbs(equ*255)
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
th2 = th2/255
th2 =  th2.astype(bool)
th2 = ~th2
light_coat = flood_fill(th2.astype(float), (0, 0),0)
dist_transform = cv2.distanceTransform(cv2.convertScaleAbs(light_coat*255),cv2.DIST_L2,5)/255.0
dist_transform = cv2.convertScaleAbs(dist_transform*255)
ret2,th2 = cv2.threshold(dist_transform,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# find contours
contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
obj = [i for i,n in enumerate(contours) if cv2.contourArea(n) > 80]



mask = np.zeros(img.shape, np.uint8)


for i in obj:
    cv2.drawContours(mask, contours, i, (255,255), -1) 

cv2.dilate(mask, kernel, 2)
sure_fg = mask.copy()/255.0


fj.conduct_watershed(test_image, sure_fg, sure_bg)




# fill holes
#sure_bg = flood_fill(th2.astype(float), (0, 0),0)






#sure_bg, sure_fg = random_walker_func(test_image)

cv2.imshow('watershed 2d', open_cv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




#stack_img = Image.open(path_obj).convert('RGB').crop((Image.open(path_obj).convert('RGB').getbbox()))
        # convert to open cv image



#convert to hsv
hsv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv_image)
# make an iterable object
zip_iter = zip(h.ravel(),s.ravel(),v.ravel())
        
# remove black pixels
chk = [i for i in zip_iter if np.sum(i) > 0]
        
# get the h, s and v vectors
h_vec = [i[0] for i in chk]
#   h_vec_app.append(h_vec)
h_parms = [np.mean(h_vec), np.std(h_vec)]
s_vec = [i[1] for i in chk]
s_parms = [np.mean(s_vec), np.std(s_vec)]
#            s_vec_app.append(s_vec)
v_vec = [i[2] for i in chk]
v_parms = [np.mean(v_vec), np.std(v_vec)]

catch = {"h_mean": h_parms[0], "h_std": h_parms[1], 
                     "s_mean": s_parms[0], "s_std": s_parms[1],"v_mean": v_parms[0], "v_std": v_parms[1]}

test_df = pd.DataFrame(catch, index = [0])
# make additional features
test_df["h_ratio"] = test_df["h_mean"]/test_df["h_std"]
test_df["s_ratio"] = test_df["s_mean"]/test_df["s_std"]
test_df["v_ratio"] = test_df["v_mean"]/test_df["v_std"]

test_df["hs_ratio"] = test_df["h_mean"]/test_df["s_mean"]
test_df["hv_ratio"] = test_df["h_mean"]/test_df["v_mean"]
test_df["sv_ratio"] = test_df["s_mean"]/test_df["v_mean"]

test_df["hsd_ratio"] = test_df["h_std"]/test_df["s_std"]
test_df["hvd_ratio"] = test_df["h_std"]/test_df["v_std"]
test_df["svd_ratio"] = test_df["s_std"]/test_df["v_std"]

# random forest model
le.inverse_transform([rf_random.best_estimator_.predict(test_df)])
# elastic  net
le.inverse_transform([lr_random.best_estimator_.predict(test_df)])
# knn
scaled_df = scaler.fit_transform(test_df)
scaled_df = pd.DataFrame(scaled_df, columns=names)
le.inverse_transform([knn_random.best_estimator_.predict(scaled_df)])
#qda
le.inverse_transform([qda_random.best_estimator_.predict(scaled_df)])
# nearest centroid
le.inverse_transform([nc_random.best_estimator_.predict(scaled_df)])

## Create the random grid
##max_depth = np.arange(1, 32)
#min_samples_split = np.linspace(0.1, 1.0, 10, endpoint=True)
#random_grid = {'n_estimators': [101],
#               "max_depth":[25],
##               'max_features': max_features,
##               'max_depth': max_depth}
#               'min_samples_split': min_samples_split}
##               'min_samples_leaf': min_samples_leaf,
##               'bootstrap': bootstrap}
#
#rf_random = GridSearchCV(estimator = rf, param_grid = random_grid, cv = 5, verbose=10, n_jobs = 6)
## Fit the random search model
#rf_random.fit(train_features, train_labels)
#rf_random.best_estimator_
#rf_random.best_score_
#rf_random.best_params_
#
#
## Create the random grid
##max_depth = np.arange(1, 32)
##min_samples_split = np.linspace(0.1, 1.0, 10, endpoint=True)
#min_samples_leaf = np.linspace(0.1, 0.5, 5, endpoint=True)
#random_grid = {'n_estimators': [101],
#               "max_depth":[25],
##               'max_features': max_features,
##               'max_depth': max_depth}
#               'min_samples_split': [0.1],
#               'min_samples_leaf': min_samples_leaf}
##               'bootstrap': bootstrap}
#
#rf_random = GridSearchCV(estimator = rf, param_grid = random_grid, cv = 5, verbose=10, n_jobs = 6)
## Fit the random search model
#rf_random.fit(train_features, train_labels)
#rf_random.best_estimator_
#rf_random.best_score_
#rf_random.best_params_
#
## Create the random grid
##max_depth = np.arange(1, 32)
##min_samples_split = np.linspace(0.1, 1.0, 10, endpoint=True)
#max_features = list(range(1,seg_beans.shape[1]))
#random_grid = {'n_estimators': [101],
#               "max_depth":[25],
#               'max_features': max_features,
##               'max_depth': max_depth}
#               'min_samples_split': [0.1],
#               'min_samples_leaf': [0.1]}
##               'bootstrap': bootstrap}
#
#rf_random = GridSearchCV(estimator = rf, param_grid = random_grid, cv = 5, verbose=10, n_jobs = 6)
## Fit the random search model
#rf_random.fit(train_features, train_labels)
#rf_random.best_estimator_
#rf_random.best_score_
#rf_random.best_params_
#
#
#
## Number of features to consider at every split
#max_features = list(range(1,seg_beans.shape[1],3))
## Maximum number of levels in tree
#max_depth = [int(x) for x in np.arange(1,32,10)]
##max_depth.append(None)
## Minimum number of samples required to split a node
#min_samples_split = np.linspace(0.1, 1.0, 10, endpoint=True)
## Minimum number of samples required at each leaf node
#min_samples_leaf = np.linspace(0.1, 0.5, 5, endpoint=True)
## Method of selecting samples for training each tree
#bootstrap = [True]




###### now try xgboost

# get a rough estimate of n_estimators
xgb1 = xgb.XGBClassifier(
 learning_rate =0.1,
 num_class = len(np.unique(train_labels)),
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 nthread=6,
 scale_pos_weight=1,
 scoring = 'accuracy',
 seed=27)

xgb_param = xgb1.get_xgb_params()

xgtrain = xgb.DMatrix(train_features, label=train_labels)

cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=1000, nfold=5,
            metrics='merror', early_stopping_rounds=50)

cvresult.shape[0]

# first tuning
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}


gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=269,
                                                      gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='accuracy',n_jobs=6,iid=False, cv=5, verbose = True)

gsearch1.fit(train_features, train_labels)
gsearch1.best_params_, gsearch1.best_score_

# second tuning
param_test2 = {
 'max_depth':[2,3,4]
 
}

gsearch2 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=269,
                                                      gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27, min_child_weight =1 ),
 param_grid = param_test2, scoring='accuracy',n_jobs=6,iid=False, cv=5, verbose = True)

gsearch2.fit(train_features, train_labels)
gsearch2.best_params_, gsearch1.best_score_

# third tuning
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}

gsearch3 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=269,
                                                      subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27, min_child_weight =1, max_depth = 3),
 param_grid = param_test3, scoring='accuracy',n_jobs=6,iid=False, cv=5, verbose = True)

gsearch3.fit(train_features, train_labels)
gsearch3.best_params_, gsearch1.best_score_

# reclaibrate n_estimators
xgb3 = xgb.XGBClassifier( learning_rate =0.1, n_estimators=269,
                                                      subsample=0.8, colsample_bytree=0.8, gamma = 0,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27, min_child_weight =1, max_depth = 3)

xgb_param = xgb3.get_xgb_params()

xgtrain = xgb.DMatrix(train_features, label=train_labels)

cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=1000, nfold=5,
            metrics='merror', early_stopping_rounds=50)

cvresult.shape[0]

 # fourth tuning
param_test4 = {
 'subsample':[i/10.0 for i in range(6,11)],
 'colsample_bytree':[i/10.0 for i in range(6,11)]
}
 

gsearch4 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=269,
                                                      subsample=0.8, colsample_bytree=0.8, gamma = 0,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27, min_child_weight =1, max_depth = 3),
 param_grid = param_test4, scoring='accuracy',n_jobs=6,iid=False, cv=5, verbose = 10)

gsearch4.fit(train_features, train_labels)
gsearch4.best_params_, gsearch4.best_score_ 
gsearch4.scoring

# fifth tuning
param_test5 = {
 'subsample':[i/100.0 for i in range(75,95,5)],
 'colsample_bytree':[i/100.0 for i in range(75,95,5)]
}
 

# do this again
gsearch5 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=269, gamma = 0,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27, min_child_weight =1, max_depth = 3),
 param_grid = param_test5, scoring='accuracy',n_jobs=6,iid=False, cv=5, verbose = 10)

gsearch5.fit(train_features, train_labels)
gsearch5.best_params_, gsearch5.best_score_ 


# sixth tuning
param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100], 
 "reg_lambda": [1e-5, 1e-2, 0.1, 1, 100]
}

gsearch6 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=269, gamma = 0,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27, min_child_weight =1, max_depth = 3, 
 subsample = 0.9, colsample_bytree = 0.9),
 param_grid = param_test6, scoring='accuracy',n_jobs=6,iid=False, cv=5, verbose = 10)

gsearch6.fit(train_features, train_labels)
gsearch6.best_params_, gsearch6.best_score_ 


# seventh tuning
param_test7 = {
 'reg_alpha':[0.001,0.01, 0.1], 
 "reg_lambda": [0.01,0.1, 1]
}

gsearch7 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=269, gamma = 0,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27, min_child_weight =1, max_depth = 3, 
 subsample = 0.9, colsample_bytree = 0.9),
 param_grid = param_test7, scoring='accuracy',n_jobs=6,iid=False, cv=5, verbose = 10)

gsearch7.fit(train_features, train_labels)
gsearch7.best_params_, gsearch7.best_score_ 

# final one 
# lastly, reduce learning rate - add more trees
param_test8 = {
 'learning_rate':[0.01]
}
 

gsearch8 = GridSearchCV(estimator = xgb.XGBClassifier(n_estimators=500, gamma = 0,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27, min_child_weight =1, max_depth = 3, 
 subsample = 0.9, colsample_bytree = 0.9, reg_alpha = 0.01, reg_lambda = 1),
 param_grid = param_test8, scoring='accuracy',n_jobs=6,iid=False, cv=5, verbose = 10)

gsearch8.fit(train_features, train_labels)
gsearch8.best_params_, gsearch8.best_score_ 

