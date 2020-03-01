# Number of features to consider at every split
import pandas as pd
import numpy as np
from PIL import Image
import cv2
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import xgboost as xgb
import sklearn.discriminant_analysis as da
from collections import Counter 
from sklearn.neighbors import NearestCentroid

le = preprocessing.LabelEncoder()

train_features = seg_beans[seg_beans.columns[1:]]
train_labels = seg_beans[seg_beans.columns[0]]
le.fit(train_labels)
train_labels = le.transform(train_labels) 
le.inverse_transform([1])
# random forest fittting and tuning

n_estimators = [int(x) for x in np.arange(1,202,10)]

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Create the random grid
random_grid = {'n_estimators': n_estimators}
#               'max_features': max_features,
#               'max_depth': max_depth,
#               'min_samples_split': min_samples_split,
#               'min_samples_leaf': min_samples_leaf,
#               'bootstrap': bootstrap}

rf_random = GridSearchCV(estimator = rf, param_grid = random_grid, cv = 5, verbose=10, n_jobs = 6)
# Fit the random search model
rf_random.fit(train_features, train_labels)
rf_random.best_estimator_
rf_random.best_score_
rf_random.best_params_





# Create the random grid
lr = rf = RandomForestClassifier()
max_depth = np.arange(1, 32)
random_grid = {'n_estimators': [81], 
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

for feature in zip(train_features.columns, rf_random.best_estimator_.feature_importances_):
    print(feature)


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

knn_random.fit(train_features, train_labels)
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
path = r"D:\Jellybean\Test_Images\Tutti-Fruitti.png"
path = r"D:\Jellybean\Test_Images\Strawberry-Jam-Signature.png"
path = r"D:\Jellybean\Test_Images\Red-Apple-Signature-ns_fg.png"
path = r"D:\Jellybean\Test_Images\Watermelon-Signature-ns_fg.png"
test_image = Image.open(path).convert('RGB').crop((Image.open(path).convert('RGB').getbbox())) 
                    
# convert to open cv image
open_cv_image = np.array(test_image)[:, :, ::-1]
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
le.inverse_transform([knn_random.best_estimator_.predict(test_df)])
#qda
le.inverse_transform([qda_random.best_estimator_.predict(test_df)])
# nearest centroid
le.inverse_transform([nc_random.best_estimator_.predict(test_df)])

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

