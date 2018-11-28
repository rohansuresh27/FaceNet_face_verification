
# coding: utf-8

# In[1]:


import time
start_time = time.time()
import cv2
import os
import glob
import pickle

import tensorflow as tf

#os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
from numpy import genfromtxt

from keras import backend as K
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')

print("--- %s seconds importing libraries ---" % (time.time() - start_time))

from fr_utils import *
from inception_blocks_v2 import *

capture_current_image = False

print("--- %s seconds to import other files ---" % (time.time() - start_time))


# In[2]:


FRmodel = faceRecoModel(input_shape=(3, 96, 96))  #(3, 196, 196)
print("--- %s seconds to load model ---" % (time.time() - start_time))

def triplet_loss(y_true, y_pred, alpha = 0.3):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
print("--- %s seconds to compile model ---" % (time.time() - start_time))

load_weights_from_FaceNet(FRmodel)
print("--- %s seconds to load weights ---" % (time.time() - start_time))


# In[3]:


# YOUR PRESENT IMAGE 

def capture_img():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("Live Frame")
    print("--- Hit Space Bar to take Image ---")

    while True:
        ret, frame = cam.read()
        cv2.imshow("Live Frame", frame)
        if not ret:
            break
        k = cv2.waitKey(1)
        
        if k%256 == 32: #remove this if image to be taken automatically
            # SPACE pressed
            img_name = "Your_present_Image_1.jpg"
            ref_img = cv2.imwrite(img_name, frame)
            print("***** Your Image Taken for Verification !! *****")
        
            print("--- Space Bar hit, closing... ---")
            cv2.destroyAllWindows()
            break
            
    your_image1 = cv2.imread('Your_present_Image_1.jpg', 1)
    face_cascade1 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face1 = face_cascade1.detectMultiScale(your_image1, 1.3, 5)  ##

    for (x,y,w,h) in face1:
        imgg = cv2.rectangle(your_image1,(x,y),(x+w,y+h),(255,255,255),2)
        roi_color1 = imgg[y:y+h, x:x+w]
        roi_color1 = cv2.resize(roi_color1, (96, 96))
        cv2.imwrite('Your_present_img.jpg',roi_color1)
        # get encoding
        current_encoding = img_to_encoding(roi_color1, FRmodel)
        
    return current_encoding


# In[5]:


#CHECK IF REFERENCE IMAGE PRESENT IN DIRECTORY, ELSE SET REF IMAGE AND ITS ENCODING

def check_ref():
    
    if os.path.isfile('C:/Users/RS_Vulcan/face_recognition/Your_ref_Image.jpg') is True:
        print("--- %s seconds checking reference image ---" % (time.time() - start_time))
        current_encoding = capture_img()
    else:
        #SET REFERENCE IMAGE
        def set_ref_img(): 
            cam = cv2.VideoCapture(0)

            cv2.namedWindow("Live Frame")
            print("--- Hit Space Bar to take Image ---")

            while True:
                ret, frame = cam.read()
                cv2.imshow("Live Frame", frame)
                if not ret:
                    break
                k = cv2.waitKey(1)

                if k%256 == 32:
                    # SPACE pressed
                    img_name = "Your_ref_Image_1.jpg"
                    ref_img = cv2.imwrite(img_name, frame)
                    print("***** Your Reference Image Set !! *****")
        
                    print("--- Space Bar hit, closing... ---")
                    cv2.destroyAllWindows()
                    break
                    
            your_image = cv2.imread('Your_ref_Image_1.jpg', 1)
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            face = face_cascade.detectMultiScale(your_image, 1.3, 5)

            for (x,y,w,h) in face:
                img = cv2.rectangle(your_image,(x,y),(x+w,y+h),(255,255,255),2)
                roi_color = img[y:y+h, x:x+w]
                roi_color = cv2.resize(roi_color, (96, 96))
                cv2.imwrite('Your_ref_Image.jpg',roi_color)
                # get encoding
                ref_data = img_to_encoding(roi_color, FRmodel)
            
            print(ref_data)
            
            # save encoding
            with open('ref_outfile', 'wb') as fp:
                pickle.dump(ref_data, fp)
                
            return print('--- Hello, Reference Image & Encoding Set, Run Code Again. ---')
        
        
        
        set_ref_img()
        
    return current_encoding


# In[16]:


def image_match():
    min_dist = 0.4
    # input reference image encoding
    with open ('ref_outfile', 'rb') as fp:
        ref_encoding = pickle.load(fp)
    # reference encoding - current encoding  
    dist = np.linalg.norm(ref_encoding - current_encoding)
    #print(dist)
    #print(ref_encoding)
    #print(current_encoding)
    
    if dist <= min_dist:
            #min_dist = dist
            print(dist)
            print('***** VERIFIED, HELLO ROHAN *****')
            
    else:
        print(dist) 
        print('***** SORRY, WRONG FACE DETECTED *****')


# In[74]:


current_encoding = check_ref()


# In[75]:


image_match()


# In[71]:


# with open('ref_outfile', 'rb') as fp:
#     ref_encoding = pickle.load(fp)
# print(ref_encoding)
    
    


# In[10]:




