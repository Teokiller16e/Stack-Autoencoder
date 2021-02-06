import matplotlib.pyplot as plt
import numpy
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ColorizationModel import createUpdatedModel

#Convert from RGB to Grayscale the dataset
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray

#Dragging input to data set and their labels:
(x_train, _), (x_test, _) = tensorflow.keras.datasets.cifar10.load_data()

#Initializing gray dataset:
xTrainGray = []
xTestGray = []

for i in x_train:
    i = rgb2gray(i)
    xTrainGray.append(i)

for i in x_test:
    i = rgb2gray(i)
    xTestGray.append(i)


#cast x_t_g:
xTrainGray = numpy.asarray(xTrainGray)
xTestGray = numpy.asarray(xTestGray)

#Normalize to float 32
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

#Normalize the gray data:
xTrainGray = xTrainGray.astype("float32")/255
xTestGray = xTestGray.astype("float32")/255

#Since we know already the shape we reshape to 1 channel:
xTrainGray = xTrainGray.reshape(xTrainGray.shape[0], 32, 32, 1)
xTestGray = xTestGray.reshape(xTestGray.shape[0], 32, 32, 1)

#Implemented Data Augmentation
datagen = ImageDataGenerator(rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
datagen.fit(x_train)

#Receive the AutoEncoderModel
finalModel = createUpdatedModel()

#Modifying Optimizer's Learning Rate:
opt = tensorflow.keras.optimizers.Adam(lr=0.01)

finalModel.compile(loss="mean_squared_error",optimizer='Adam',metrics=['accuracy'])
finalModel.summary()

history = finalModel.fit(xTrainGray,x_train,batch_size=32,epochs=10,validation_data=(xTestGray,x_test))

# #Final Accuracy
scores = finalModel.evaluate(xTestGray,x_test,batch_size=32)
print("Model Accuracy : %.2f%%"%(scores[1]*100))

#Plot Loss
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'black',linewidth = 3.0)
plt.plot(history.history['val_loss'],'black',ls='--',linewidth = 3.0)
plt.legend(['Training Loss' , 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs' ,fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
plt.show()

#Plot Accuracy:

plt.figure(figsize=[8,6])
plt.plot(history.history['accuracy'],'black',linewidth = 3.0)
plt.plot(history.history['val_accuracy'],'black',ls='--',linewidth = 3.0)
plt.legend(['Training Accuracy' , 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs' ,fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.show()


#Final prediction results
outputImages = finalModel.predict(xTestGray)

tes = plt.figure(1)
for i in range (5):
    tes.add_subplot(1,5,i+1)
    plt.imshow(xTestGray[i],cmap='gray')


fin = plt.figure(2)
for i in range (5):
    fin.add_subplot(1,5,i+1)
    plt.imshow(outputImages[i])

plt.show()