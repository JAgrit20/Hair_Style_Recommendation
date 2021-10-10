
import requests
from matplotlib.pyplot import imshow
import time
from PIL import Image, ImageDraw
import face_recognition
import pandas as pd
import numpy as np
from os.path import basename
import math
import pathlib
from pathlib import Path
import os
import random
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA 


image_dir = "data/pics"   #celebrity search version



def distance(p1,p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx*dx+dy*dy)

def scale_rotate_translate(image, angle, center = None, new_center = None, scale = None, resample=Image.BICUBIC):
    if (scale is None) and (center is None):
        return image.rotate(angle=angle, resample=resample)
    nx,ny = x,y = center
    sx=sy=1.0
    if new_center:
        (nx,ny) = new_center
    if scale:
        (sx,sy) = (scale, scale)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine/sx
    b = sine/sx
    c = x-nx*a-ny*b
    d = -sine/sy
    e = cosine/sy
    f = y-nx*d-ny*e
    return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)

def crop_face(image, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.3,0.3), dest_sz = (600,600)):
    # calculate offsets in original image
    offset_h = math.floor(float(offset_pct[0])*dest_sz[0])
    offset_v = math.floor(float(offset_pct[1])*dest_sz[1])
    # get the direction
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians
    rotation = -math.atan2(float(eye_direction[1]),float(eye_direction[0]))
    # distance between them
    dist = distance(eye_left, eye_right)
    # calculate the reference eye-width
    reference = dest_sz[0] - 2.0*offset_h
    # scale factor
    scale = float(dist)/float(reference)
    # rotate original around the left eye

    image = scale_rotate_translate(image, center=eye_left, angle=rotation)
    # crop the rotated image
    crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
    crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
    image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
    # resize it
    image = image.resize(dest_sz, Image.ANTIALIAS)
    return image

def make_face_df_save(image_select,filenum,df):
    
    # This function looks at one image, draws points and saves points to DF
    pts = []
   # filenum = 0   # need this to iterate through the dataframe to append rows
    face = 0
    image = face_recognition.load_image_file(image_select)
    face_landmarks_list = face_recognition.face_landmarks(image)
    
    for face_landmarks in face_landmarks_list:
        face += 1
        if face >1:    # this will only measure one face per image
            break
        else:
            # Print the location of each facial feature in this image
            facial_features = [
                'chin',
                'left_eyebrow',
                'right_eyebrow',
                'nose_bridge',
                'nose_tip',
                'left_eye',
                'right_eye',
                'top_lip',
                'bottom_lip'
                ]

            for facial_feature in facial_features:
                # put each point in a COLUMN
                for  point in  face_landmarks[facial_feature]:
                    for pix in point:
                        pts.append(pix)
               
        pil_image = Image.fromarray(image)
        d = ImageDraw.Draw(pil_image)
        
        eyes = []
        lex = pts[72]
        ley = pts[73]
        rex = pts[90]
        rey = pts[91]
        eyes.append(pts[72:74])
        eyes.append(pts[90:92])

        image =  Image.open(image_select)
        crop_image = crop_face(image, eye_left=(lex, ley), eye_right=(rex, rey), offset_pct=(0.34,0.34), dest_sz=(300,300))
        try:
            crop_image.save(str(image_select)+"_NEW_cropped.jpg")
        except:
            continue
        #crop_image.show()
        
        nn = str(image_select)+"_NEW_cropped.jpg"
        pts = []
        face = 0
        image = face_recognition.load_image_file(nn)
        face_landmarks_list = face_recognition.face_landmarks(image)

        for face_landmarks in face_landmarks_list:
            face += 1
            if face >1:    # this will only measure one face per image
                break
            else:
                # Print the location of each facial feature in this image
                facial_features2 = [
                    'chin',
                    'left_eyebrow',
                    'right_eyebrow',
                    'nose_bridge',
                    'nose_tip',
                    'left_eye',
                    'right_eye',
                    'top_lip',
                    'bottom_lip'
                    ]

                for facial_feature in facial_features2:
                    # put each point in a COLUMN
                    for  point in  face_landmarks[facial_feature]:
                        for pix in point:
                            pts.append(pix)

            i = 0
            for j in range(0,17):
                if i != 16:
                    if i != 17:
                        px = pts[i]
                        py = pts[i+1]
                        chin_x = pts[16]   # always the chin x
                        chin_y = pts[17]   # always the chin y

                        x_diff = float(px - chin_x)

                        if(py == chin_y): 
                            y_diff = 0.1
                        if(py < chin_y): 
                            y_diff = float(np.absolute(py-chin_y))
                        if(py > chin_y):
                            y_diff = 0.1
                            print("Error: facial feature is located below the chin.")

                        angle = np.absolute(math.degrees(math.atan(x_diff/y_diff)))

                        pts.append(angle)
                i += 2
        
            pil_image = Image.fromarray(image)
            d = ImageDraw.Draw(pil_image)

            for facial_feature in facial_features2:
                    #d.line(face_landmarks[facial_feature], width=5)
                    d.point(face_landmarks[facial_feature], fill = (255,255,255))
            
            
            pil_image.save(str(image_select) + '_NEW_rotated_pts.jpg', 'JPEG', quality = 100)

            # take_measurements width & height measurements
        msmt = []
        a = pts[0]   ## point 1 x - left side of face 
        b = pts[1]   ## point 1 y
        c = pts[32]  ## point 17 x - right side of face
        d = pts[33]  ## point 17 y

        e = pts[16]  ## point 9 x - chin
        f = pts[17]  ## point 9 y - chin
        #Visual inspection indicates that point 29 is the middle of the face, 
        #so the height of the face is 2X the height between chin & point 29 which are coordinates 56 and 57     
        g = pts[56]  # point 29's x coordinate (mid-face point)
        h = pts[57]   # point 29's y coordinate
        
        i = pts[12]    # point 7 x   for jaw length 
        j = pts[13]    # point 7 y   for jaw length
        k = pts[20]    # point 11 x  for jaw length
        l = pts[21]    # point 11 y  for jaw length
             
        m = pts[8]     # point 5 x   for lower jaw length     
        n = pts[9]     # point 5 y
        o = pts[24]     # point 13 x
        p = pts[25]     # point 13 y

        face_width = np.sqrt(np.square(a - c) + np.square(b - d))
        #print(face_width)
        pts.append(face_width)
        face_height = np.sqrt(np.square(e - g) + np.square(f - h)) * 2   # double the height to the mid-point
        #print(face_height)
        pts.append(face_height)
        height_to_width = face_height/face_width
        
        pts.append(height_to_width)
        
        # JAW width (7-11)
        jaw_width = np.sqrt(np.square(i-k) + np.square(j-l))
        pts.append(jaw_width)
        jaw_width_to_face_width =  jaw_width/face_width
        pts.append(jaw_width_to_face_width)
        
        # mid-JAW width (5-13)
        mid_jaw_width = np.sqrt(np.square(m-o) + np.square(n-p))
        pts.append(mid_jaw_width)
        mid_jaw_width_to_jaw_width =  mid_jaw_width/jaw_width
        pts.append(mid_jaw_width_to_jaw_width)
        
        ### end of new ###
            
        df.loc[filenum] = np.array(pts)
        #imshow(pil_image, cmap='gray')
        
            
def find_face_shape(df,file_num):
    data = pd.read_csv('all_features.csv',index_col = None)
    data = data.drop('Unnamed: 0',axis = 1)

    data_clean = data.dropna(axis=0, how='any')
    X = data_clean
    X = X.drop(['filenum','filename','classified_shape'] , axis = 1)
    X_norm = normalize(X)
    Y = data_clean['classified_shape']

    scaler = StandardScaler()  
    scaler.fit(X)  

    X = scaler.transform(X)

    ### Split into train/test sets

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,Y,
        test_size=0.25,
        random_state=1200)

    ### Apply PCA for dimension reduction

    n_components = 18
    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True).fit(X)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # #Remove PCA 
    X_train_pca = X_train
    X_test_pca = X_test

    ## Neural Network (MLP)

    from sklearn.neural_network import MLPClassifier

    # With best model tuning

    best_mlp = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(60, 100, 30, 100), learning_rate='constant',
           learning_rate_init=0.01, max_iter=100, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=525,
           shuffle=True, solver='sgd', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=False)
    best_mlp.fit(X_train_pca, Y_train)

    mlp_score = best_mlp.score(X_test_pca,Y_test)

    y_pred = best_mlp.predict(X_test_pca)

    mlp_crosstab = pd.crosstab(Y_test, y_pred, margins=True)
    
    test_row = df.loc[file_num].values.reshape(1,-1)
    test_row = scaler.transform(test_row)  
    test_shape = best_mlp.predict(test_row)
    return test_shape

def getting_image_attribute():
    # file_num = 2035
    
    df2 = pd.DataFrame(columns = ['0','1','2','3','4','5','6','7','8','9','10','11',	'12',	'13',	'14',	'15',	'16','17',
                                '18',	'19',	'20',	'21',	'22',	'23',	'24','25',	'26',	'27',	'28',	'29',
                                '30',	'31',	'32',	'33',	'34',	'35',	'36',	'37',	'38',	'39',	'40',	'41',
                                '42',	'43',	'44',	'45',	'46',	'47',	'48',	'49',	'50',	'51',	'52',	'53',
                                '54',	'55',	'56',	'57',	'58',	'59',	'60',	'61',	'62',	'63',	'64',	'65',
                                '66',	'67',	'68',	'69',	'70',	'71',	'72',	'73',	'74',	'75',	'76',	'77',
                                '78',	'79',	'80',	'81',	'82',	'83',	'84',	'85',	'86',	'87',	'88',	'89',
                                '90',	'91',	'92',	'93',	'94',	'95',	'96',	'97',	'98',	'99',	'100',	'101',
                                '102',	'103',	'104',	'105',	'106',	'107',	'108',	'109',	'110',	'111',	'112',	'113',
                                '114',	'115',	'116',	'117',	'118',	'119',	'120',	'121',	'122',	'123',	'124',	'125',
                                '126',	'127',	'128',	'129',	'130',	'131',	'132',	'133',	'134',	'135',	'136',	'137',
                                '138',	'139',	'140',	'141',	'142',	'143','A1','A2','A3','A4','A5','A6','A7','A8','A9',
                                'A10','A11','A12','A13','A14','A15','A16','Width','Height','H_W_Ratio','Jaw_width','J_F_Ratio',
                                'MJ_width','MJ_J_width'])
    
    print(df2)

    import os, sys

# Open a file
    path = "A:/Hairstyle_recommender/Hair_Style_Recommendation/shots2/"
    dirs = os.listdir( path )

# This would print all the files and directorie
    i =0
    shapes = []
    for file in dirs:
        # print("file in the upload directory", file)
        newf = path+ file
        make_face_df_save(newf, i, df2)
        print(df2)
        
        shapes.append(find_face_shape(df2, i))
        i= i+1
        # df2.append(df1)
    df2["image"] = np.array(dirs)
    df2["classified_shape"] = np.array(shapes)
    # directory = "Mens_picture_data"
    # parent_dir = "A:/Hairstyle_recommender/Hair_Style_Recommendation/"
    # path = os.path.join(parent_dir, directory)
    # print("df2", df2)
    # os.mkdir(path)
    
    # parent_dir = "A:/Hairstyle_recommender/Hair_Style_Recommendation/Mens_picture_data/"
    # directory_list = ["oval", "round", "heart", "square", "long"]
    # for directory in directory_list:
    #     path = os.path.join(parent_dir, directory)
    #     os.mkdir(path)
    df2.to_csv('check.csv')  
    
    total_pictures = df2.shape[0]
    image_path_initial = list(df2["image"])[0]
    print("image_path", image_path_initial)
    image =  Image.open(path+image_path_initial)
    image_save_path = "A:/Hairstyle_recommender/Hair_Style_Recommendation/Mens_picture_data/"+str(list(df2["classified_shape"])[0])+ '/'+ image_path_initial
    print(image_save_path)
    image.save(image_save_path)
    imshow(image)    
    # for i in range(total_pictures):
    #     image_path_initial = df2["image"]
    #     image =  Image.open(image_path_initial)
    #     image.save("A:/Hairstyle_recommender/Hair_Style_Recommendation/Mens_picture_data/"+df2[i]["classified_shape"]+"/"+str(i)+".jpg")
        

    # df2.to_csv('check.csv')  
    print(df2)
    # df2.to_csv('A:/Hairstyle_recommender/Hair_Style_Recommendation/df_test.csv')
    
getting_image_attribute()
