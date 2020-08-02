import numpy as np
import cv2
import os
from keras.models import load_model
import pickle as pkl
from PIL import Image
import pygame
from keras.applications.resnet50 import preprocess_input
pygame.init()
screen = pygame.display.set_mode((200,200),pygame.RESIZABLE)

CLIP_X1 = 150
CLIP_Y1 = 140
CLIP_X2 = 500
CLIP_Y2 = 460

model = pkl.load(open("naruto_me.pickle","rb"))

cap = cv2.VideoCapture(0)

while True:
    _, FrameImage = cap.read()
    FrameImage = cv2.flip(FrameImage, 1)
    FrameImage = cv2.rectangle(FrameImage, (CLIP_X1, CLIP_Y1), (CLIP_X2, CLIP_Y2), (0,255,0) ,3)
    
    

    crop = FrameImage[CLIP_Y1:CLIP_Y2,CLIP_X1:CLIP_X2]
    crop = cv2.resize(crop,(224,224))
    ROI = cv2.cvtColor(FrameImage, cv2.COLOR_BGR2RGB)

    cv2.imshow("ROI",FrameImage)
    
    categories = ['\iboar','\dog','\horse','\monkey','\snake','\itiger','\irat']

    try:
        result  = categories[np.argmax(model.predict(preprocess_input(crop.reshape(1, 224, 224, 3))))]
        predict_img  = pygame.image.load(os.getcwd() + result + '_disp.jpg')
    except IndexError:
        predict_img  = pygame.image.load(os.getcwd() + '\nosign.png')
        
    
    
    predict_img = pygame.transform.scale(predict_img, (200, 200))
    screen.blit(predict_img, (0,0))
    pygame.display.flip()
    interrupt = cv2.waitKey(10)

    if interrupt & 0xFF == ord('q'): # esc key
        break
            
pygame.quit()
cap.release()
cv2.destroyAllWindows()