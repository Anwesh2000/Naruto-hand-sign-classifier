import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.activations import relu,softmax
import matplotlib.pyplot as plt
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
import pickle as pkl
from keras.layers import Conv2D, MaxPool2D, Activation, Dropout,Dense,GlobalAveragePooling2D
from keras.models import Sequential,Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.applications.resnet50 import ResNet50, preprocess_input

NAME = "128_128_layer_dropout_0.5"

pickle_in = open("/content/drive/My Drive/Naruto hand sign detector/X_test_rgb.pickle","rb")
X_test = pkl.load(pickle_in)

pickle_in = open("/content/drive/My Drive/Naruto hand sign detector/Y_test_rgb.pickle","rb")
Y_test = pkl.load(pickle_in)

pickle_in = open("/content/drive/My Drive/Naruto hand sign detector/X_train_me.pickle","rb")
X_train = pkl.load(pickle_in)

pickle_in = open("/content/drive/My Drive/Naruto hand sign detector/Y_train_me.pickle","rb")
Y_train = pkl.load(pickle_in)

# one hot encodeing Y data
Y_test = np.array(Y_test).reshape(-1,1)
enc = OneHotEncoder(handle_unknown ='ignore')
Y_test = enc.fit_transform(Y_test)

Y_train = np.array(Y_train).reshape(-1,1)
enc = OneHotEncoder(handle_unknown ='ignore')
Y_train = enc.fit_transform(Y_train)

# preprocessing X data for resnet
X_test =preprocess_input(X_test)
X_train = preprocess_input(X_train)

# resnet50 model
base_model = ResNet50(weights='imagenet', 
                      include_top=False, 
                      input_shape= X_train.shape[1:])

# freezing the weights of ResNet model except the last 2 convolution blocks
for layer in base_model.layers[:-8]:
  layer.trainable = False

last_layer = base_model.output
# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(last_layer)
# add fully-connected & dropout layers

x = Dense(128, activation='relu',name='fc-1')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu',name='fc-2')(x)
x = Dropout(0.5)(x)

# a softmax layer for 7 classes
out = Dense(7, activation='softmax',name='output_layer')(x)
finetune_model = Model(inputs=base_model.input, outputs= out)


tensorboard = TensorBoard(log_dir="logs/{}".format(NAME),histogram_freq= 1)# to visualise the training of model


finetune_model.compile(optimizer=Adam(learning_rate = 0.001),loss= 'categorical_crossentropy',metrics= ['accuracy'])

finetune_model.fit(X_train,Y_train, batch_size = 64, epochs =100,validation_data= (X_test,Y_test),callbacks=[tensorboard], workers= 10)



pkl.dump(finetune_model,open("model.pickle","wb"))
