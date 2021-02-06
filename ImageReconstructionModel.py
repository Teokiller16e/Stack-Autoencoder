import numpy
import tensorflow
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization
from tensorflow.keras.models import Sequential

#BASE ARCHITECTURE FIRST EXERCISE
def createModel():
    inputShape = (32,32,3)
    model = Sequential()
    # Construction of ConvolutionLayer & MaxPooling set :
    model.add(Conv2D(8,(3,3), padding='same',input_shape = inputShape,strides=2))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(padding='same',pool_size=(2,2)))

    model.add(Conv2D(12,(3,3), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(padding='same',pool_size=(2,2)))
    #model.add(Dropout(0.4))

    model.add(Conv2D(16,(3,3), padding='same'))
    model.add(Activation("relu"))
    model.add(UpSampling2D(size=(2,2)))

    model.add(Conv2D(12,(3,3), padding='same'))
    model.add(Activation("relu"))
    model.add(UpSampling2D(size=(4,4)))

    model.add(Conv2D(3,(1,1), padding='same'))
    model.add(Activation('sigmoid'))
    return model

#MODIFIED ACRCHITECTURE FIRST EXECRCISE
def improvedCreateModel():
    inputShape = (32,32,3)
    model = Sequential()
    # Construction of ConvolutionLayer & MaxPooling set :
    model.add(Conv2D(16,(3,3), padding='same',input_shape = inputShape,strides=1))
    model.add(Activation("relu"))
    #model.add(MaxPooling2D(padding='same',pool_size=(2,2)))

    model.add(Conv2D(32,(3,3), padding='same',strides=1))
    model.add(Activation("relu"))
    #model.add(MaxPooling2D(padding='same',pool_size=(2,2)))

    model.add(Conv2D(64,(3,3), padding='same'))
    model.add(Activation("relu"))
    #model.add(UpSampling2D(size=(2,2)))

    model.add(Conv2D(64,(3,3), padding='same'))
    model.add(Activation("relu"))
    #model.add(UpSampling2D(size=(2,2)))

    model.add(Conv2D(32,(3,3),padding='same'))

    model.add(Conv2D(3,(3,3), padding='same'))
    model.add(Activation('sigmoid'))
    return model

