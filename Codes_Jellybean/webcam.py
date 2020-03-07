import cv2
import sys
from scipy import ndimage
import numpy as np
import pickle
import pandas as pd
#cascPath = sys.argv[1]
#faceCascade = cv2.CascadeClassifier(cascPath)

cam = cv2.VideoCapture(0)
le = pickle.load(open(r"D:\Jellybean\le.sav", 'rb'))
rf_random = pickle.load(open(r"D:\Jellybean\random_forest.sav", 'rb'))
img_counter = 0
while True:
    ret, frame = cam.read()
#    print(ret)
    cv2.imshow("image", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
#        cv2.imwrite(img_name, frame) 
#        print("{} written!".format(img_name))
        img_counter += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        test_image = frame.copy()
#    path = r"D:\Jellybean\Test_Images\pomegranate-jelly-belly-beans.png"
        test_image = gray.copy()
        ret2,th2 = cv2.threshold(test_image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
        th2 = cv2.convertScaleAbs(ndimage.binary_fill_holes(th2).astype(float)*255)
        mask_3d = np.dstack([th2.astype(bool)]*3)
        test_image = frame.copy()
        new_img = test_image*mask_3d
        open_cv_image = new_img.copy()
    
    
        h, s, v = cv2.split(open_cv_image)
        print(h.shape)
        print(type(h))
#    name = ["pomogranate"]
        
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
    
        pred = le.inverse_transform([rf_random.best_estimator_.predict(seg_beans_rgb)])
        print(pred)
        cv2.putText(frame,pred[0], (100,100), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
        cv2.imshow("image",frame)
#        cv2.imshow("Video",open_cv_image)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
##    faces = faceCascade.detectMultiScale(
##        gray,
##        scaleFactor=1.1,
##        minNeighbors=5,
##        minSize=(30, 30),
##        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
##    )
#
#    # Draw a rectangle around the faces
##    for (x, y, w, h) in faces:
##        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#
#    # Display the resulting frame
#        cv2.imshow('Video', frame)

# When everything is done, release the capture
cam.release()

cv2.destroyAllWindows()