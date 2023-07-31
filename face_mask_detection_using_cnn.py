#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
#from google.colab.patches import cv2_imshow
from PIL import Image 
from sklearn.model_selection import train_test_split




with_mask_files = os.listdir("C:\\Users\\psykid\\Desktop\\AI PHASE\\datasets\\data\\with_mask")
print(with_mask_files[0:5])
print(with_mask_files[-5:])



without_mask_files = os.listdir("C:\\Users\\psykid\\Desktop\\AI PHASE\\datasets\\data\\without_mask")
print(without_mask_files[0:5])
print(without_mask_files[-5:])



print('Number of with mask images:', len(with_mask_files))
print('Number of without mask images:', len(without_mask_files))


# Creating Labels for the two class of Images
# 
# with mask --> 1
# 
# without mask --> 0



with_mask_labels = [1]*3725
without_mask_labels = [0]*3828
print(with_mask_labels[0:5])
print(without_mask_labels[0:5])



print(len(with_mask_labels))
print(len(without_mask_labels))



labels = with_mask_labels + without_mask_labels

print(len(labels))
print(labels[0:5])
print(labels[-5:])



# displaying with mask image
img = mpimg.imread("C:\\Users\\psykid\\Desktop\\AI PHASE\\datasets\\data\\with_mask\\with_mask_1335.jpg")
imgplot = plt.imshow(img)
plt.show()



# displaying without mask image
img = mpimg.imread("C:\\Users\\psykid\\Desktop\\AI PHASE\\datasets\\data\\without_mask\\without_mask_2925.jpg")
imgplot = plt.imshow(img)
plt.show()


# Image Processing


# convert images to numpy arrays+

with_mask_path ="C:\\Users\\psykid\\Desktop\\AI PHASE\\datasets\\data\\with_mask"

data = []

for img_file in with_mask_files:

  image = Image.open(with_mask_path +'\\'+ img_file)
  image = image.resize((128,128))
  image = image.convert('RGB')
  image = np.array(image)
  data.append(image)



without_mask_path = "C:\\Users\\psykid\\Desktop\\AI PHASE\\datasets\\data\\without_mask"

for img_file in without_mask_files:

  image = Image.open(without_mask_path +'\\'+ img_file)
  image = image.resize((128,128))
  image = image.convert('RGB')
  image = np.array(image)
  data.append(image)


print(type(data))
print(len(data))

data[0]





type(data[0])


data[0].shape


# converting image list and label list to numpy arrays

X = np.array(data)
Y = np.array(labels)


print(X.shape)
print(Y.shape)

type(X)
type(Y)


# Train Test Split


X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


print(X.shape, X_train.shape, X_test.shape)


# scaling the data

X_train_scaled = X_train/255

X_test_scaled = X_test/255


X_train[0]


X_train_scaled[0]


# Building a Convolutional Neural Networks (CNN)


import tensorflow as tf
from tensorflow import keras

num_of_classes = 2

model = keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))


model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))


model.add(keras.layers.Dense(num_of_classes, activation='sigmoid'))





# compile the neural network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])





# training the neural network
history = model.fit(X_train_scaled, Y_train, validation_split=0.1, epochs=5)


# Model Evaluation

loss, accuracy = model.evaluate(X_test_scaled, Y_test)
print('Test Accuracy =', accuracy)




h = history

# plot the loss value
plt.plot(h.history['loss'], label='train loss')
plt.plot(h.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

# plot the accuracy value
plt.plot(h.history['acc'], label='train accuracy')
plt.plot(h.history['val_acc'], label='validation accuracy')
plt.legend()
plt.show()




  
