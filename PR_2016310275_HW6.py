import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers

studentID = 2016310275
np.random.seed(studentID)

dataGen = ImageDataGenerator(rescale=1./255,
                         rotation_range=20,
                         width_shift_range=0.1,
                         height_shift_range= 0.1,
                         shear_range=0.1,
                         zoom_range=0.1,
                         horizontal_flip=True,
                         fill_mode='nearest')
trainGen = dataGen.flow_from_directory(directory='./train', target_size=(150, 150), batch_size=20)
valDataGen = ImageDataGenerator(rescale=1./255,
                         rotation_range=20,
                         width_shift_range=0.1,
                         height_shift_range= 0.1,
                         shear_range=0.1,
                         zoom_range=0.1,
                         horizontal_flip=True,
                         fill_mode='nearest')
validationGen = valDataGen.flow_from_directory(directory='./validation', target_size=(150, 150), batch_size=20)

model = Sequential()
pre_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
pre_model.trainable = False
model.add(pre_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
pre_model.summary()
model.summary()

model.compile(loss= 'binary_crossentropy', optimizer=keras.optimizers.SGD(learning_rate=2e-5))

# batch size = 20,
# trainSet = 2000 -> steps_per_epoch = 100
# validation dataSet = 1000 -> validation_steps = 50
model.fit_generator(trainGen,
                    steps_per_epoch=100,
                    validation_data=validationGen,
                    validation_steps=50,
                    epochs=10)

scores = model.evaluate_generator(validationGen, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))