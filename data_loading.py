import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2
import random
import pandas as pd
import pickle
from random import randint
from PIL import Image

# converts images to numpy arrays and creates X and Y data

def create_data(data_dir,hand_sign_list,):
    training_data = []
    for hand_sign in hand_sign_list:
        path = os.path.join(data_dir,hand_sign)
        classnum = hand_sign_list.index(hand_sign)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img))
            rgb_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            new_array = cv2.resize(rgb_array,(224,224)) #input dimension for ResNet50
            training_data.append([new_array,classnum])
    random.shuffle(training_data) #shuffles the data set
    training_data = np.array(training_data)
    X = []
    y = []

    for features,label in training_data:
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, 224, 224, 3)
    
    
    X_file_name = 'X_'+ os.path.basename(data_dir) + '.pickle'
    Y_file_name = 'Y_'+ os.path.basename(data_dir) + '.pickle'
    
    pickle_out = open(X_file_name,"wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()
    
    pickle_out = open(Y_file_name,"wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

categories = ['boar','dog','horse','monkey','snake','tiger','rat']

create_data("Naruto Hand Sign Data/train",categories) #creation of training data
create_data("Naruto Hand Sign Data/test",categories)  #creation of testing data

    