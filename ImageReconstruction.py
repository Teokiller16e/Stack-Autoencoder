import numpy
import matplotlib
import matplotlib.pyplot as plt
import tensorflow
from ImageReconstructionModel import createModel
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dragging input to data set and their labels:
(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.cifar10.load_data()

# Convert to float 32
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Normalizing input(Works-->Tested) :
x_train = x_train / 255
x_test = x_test / 255

# Merging data xTrain & xTest
input = numpy.concatenate((x_train, x_test), axis=0)
labels = numpy.concatenate((y_train, y_test), axis=0)

# #splitting data 1 to training and testing
x_train, x_test, y_train, y_test = train_test_split(input, labels, random_state=0, test_size=0.2)

optimizer = tensorflow.keras.optimizers.Adam()
# optimizer = keras.optimizers.Adam(lr=0.01)

# Obtaining the model from model class :
finalModel = createModel()
finalModel.compile(loss="mse", optimizer='Adam', metrics=['accuracy'])
finalModel.summary()

# Data augmentation:
datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, rescale=1. / 255,
                             shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
datagen.fit(x_train)

# Assigning the model.fit to history variable for plotting the acc/loss of the model:
history = finalModel.fit(x_train, x_train, batch_size=32, epochs=10, validation_split=0.5, validation_data=(x_test, x_test))

# Final Accuracy
scores = finalModel.evaluate(x_test, x_test, batch_size=32)
print("Model Accuracy : %.2f%%" % (scores[1] * 100))

# Plot Loss & Accuracy Curves
# Plot Loss :
plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'], 'black', linewidth=3.0)
plt.plot(history.history['val_loss'], 'black', ls='--', linewidth=3.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)
plt.show()

# Plot Accuracy:

plt.figure(figsize=[8, 6])
plt.plot(history.history['accuracy'], 'black', linewidth=3.0)
plt.plot(history.history['val_accuracy'], 'black', ls='--', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)
plt.show()

# Insert predictions to object and plot between x_test's and predictions (visualized comparison)
outputImages = finalModel.predict(x_test)

for i in range(5):
    plt.imshow(x_test[i])
    plt.show()

    plt.imshow(outputImages[i])
    plt.show()

# run the model with xTest
# predictions received
# visually comparison: true xTrain - predictions
# use 11 convert input to grayscale images and after (xTrain,yTrain)
# loss , accuracy = finalModel.evaluate(xTest,yTest)
