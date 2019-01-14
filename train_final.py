# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eVBVbK-NLhzG45qQjH9XFdtc1Vyq6BWQ
"""

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.layers.advanced_activations import LeakyReLU

img_width, img_height = 256, 256

model = applications.ResNet50(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

x = model.output
x = GlobalAveragePooling2D()(x)
# x = Flatten()(x)
x = Dense(1024)(x) #, activation="relu")(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(0.5)(x)
x = Dense(1024)(x) #, activation="relu")(x)
x = LeakyReLU(alpha=0.2)(x)
predictions = Dense(5, activation="softmax")(x)

model_final = Model(input = model.input, output = predictions)
model_final.load_weights("/content/gdrive/My Drive/resnet50_train.hdf5")
# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.Adam(lr=0.0001, decay=0.9), metrics=["accuracy"])

ls cnn2/Data/train/dry_cans/

train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

valid_datagen = ImageDataGenerator(
rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
'cnn2/Data/train/',
target_size = (256, 256),
batch_size = 10, 
class_mode = "categorical")

class_indices = train_generator.class_indices
print(train_generator.class_indices)

import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

import cv2
import glob


filenames = []
for filename in glob.iglob("cnn2/Data/test/*.jpg"):
  filenames.append(filename)


  
import pandas
classes = ['dry_cans', 'dry_other', 'dry_paper', 'dry_plastic', 'wet']

final_pred_classes = []
final_names = []

for i in range(1108):
    print(i)
    img=image.load_img(filenames[i], target_size=(256,256))
    x = image.img_to_array(img)
    x = x.astype('float32') / 255
    x = np.expand_dims(x, axis=0)

    pred = model_final.predict(x)
    pred_label = np.argmax(pred,1)
    class_name = classes[pred_label[0]]
#     print()
    new_filename = filenames[i].split('/')
    final_names.append(new_filename[3])
    
#     print(new_filename[3], class_name)
    final_pred_classes.append(class_name)
    
a= pandas.DataFrame({"Filename":final_names,"Category":final_pred_classes})
a.to_csv("submission_new.csv",index=False)



from google.colab import files
files.download('submission.csv')
