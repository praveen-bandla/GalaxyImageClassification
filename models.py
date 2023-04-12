'''
Model defintions for all models used in this project

includes:
- VGG_16()
- VGG_19()
- CNN_implementation()

'''

#import libraries 
import tensorflow as tf
from tensorflow import keras

from keras import layers
from keras.models import Sequential
from keras.layers import Dropout, Dense, Conv2D, MaxPool2D , Flatten, Activation, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop


#Model 
def VGG_16():
  '''
  Implementing VGG_16's pre-existing architecture
  Using definition provided here:
  https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
  '''
  model = Sequential()
  input_shape = (106,106,3)
  model.add(Conv2D(input_shape=input_shape,filters=64,kernel_size=(3,3),padding="same", activation="relu"))

  model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

  model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

  model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

  model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

  model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

  model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

  model.add(Flatten())
  model.add(Dense(units=4096,activation="relu"))
  model.add(Dropout(0.5))
  model.add(Dense(units=4096,activation="relu"))
  model.add(Dropout(0.5))
  #using sigmoid as final activation function
  model.add(Dense(units=37, activation="sigmoid"))
  
  #returning the model
  return model



#Model 
def VGG_19():
  '''
  Implementing VGG_19's pre-existing architecture
  Using definition provided here:
  https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
  '''
  model = Sequential()
  input_shape = (106,106,3)

  model.add(Conv2D(input_shape=input_shape,filters=64,kernel_size=(3,3),padding="same", activation="relu"))

  model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
  
  model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
  
  model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
  
  model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
  
  model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
  
  model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
  
  model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
  
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
  
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
  
  model.add(Flatten())
  model.add(Dense(units=4096,activation="relu"))
  model.add(Dropout(0.5))
  model.add(Dense(units=4096,activation="relu"))
  model.add(Dropout(0.5))
  #using softmax as final activation function
  model.add(Dense(units=37, activation="softmax"))

  return model


def CNN_implementation():
    '''
    Implemnting a version of a CNN
    Inspired from:
    https://jayspeidell.github.io/portfolio/project07-galazy-zoo/
    '''

    model = Sequential()
    input_shape = (106, 106, 3)
    model.add(Conv2D(input_shape = input_shape, filters = 64, kernel_size = (3,3), padding = "same"))
    model.add(layers.Activation("relu"))
    model.add(MaxPool2D(pool_size = (2,2),strides = (2,2)))

    model.add(Conv2D(filters = 96, kernel_size = (3,3), strides = 1, padding = "same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(MaxPool2D(pool_size = (2,2),strides = (2,2)))

    model.add(Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = "same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(MaxPool2D(pool_size = (2,2),strides = (2,2)))

    model.add(Flatten())
    model.add(Dense(units = 1024, activation = "relu"))
    model.add(Dropout(0.15))
    model.add(Dense(units = 256, activation = "relu"))
    model.add(Dropout(0.15))
    #using sigmoid as final activation function
    model.add(Dense(units = 37, activation = "sigmoid"))

    #returning the model
    return model
