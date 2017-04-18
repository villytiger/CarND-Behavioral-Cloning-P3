from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sklearn


# Read csv files from all subdirectories of 'data' directory.
# This allows to simply add new training data.

samples = []
for dir_name in os.listdir('data'):
    with open(os.path.join('data', dir_name, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            for sample in zip(line[:3], [0, 0.2, -0.2]):
                sample_path = os.path.join('data', dir_name, 'IMG', sample[0].split('\\')[-1])
                sample_angle = float(line[3]) + sample[1]
                samples.append({'path': sample_path, 'angle' : sample_angle, 'invert': False})

train_samples, valid_samples = train_test_split(samples, test_size=0.2)

# Augment training data by inverting image and measurement

augmented_samples = []
for sample in train_samples:
    augmented_samples.append(sample)
    
    sample = sample.copy()
    sample['invert'] = True
    augmented_samples.append(sample)

def generator(samples, batch_size=256):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample['path'])
                center_angle = batch_sample['angle']
                if sample['invert']:
                    center_image = np.fliplr(center_image)
                    center_angle = -center_angle
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)        

train_generator = generator(augmented_samples)
train_steps = len(augmented_samples) / 256
            
valid_generator = generator(valid_samples)
valid_steps = len(valid_samples) / 256

# Use Nvidia neural network architecture

model = Sequential()

model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(BatchNormalization())

model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))

model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(Conv2D(64, (3, 3), activation='elu'))

model.add(Flatten())

model.add(Dropout(0.3))
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')


early_stopping = EarlyStopping(patience=0, verbose=1)

history_object = model.fit_generator(train_generator, train_steps, validation_data=valid_generator,
                                     validation_steps=valid_steps, epochs=3,
                                     callbacks=[early_stopping])

model.save('model.h5')
