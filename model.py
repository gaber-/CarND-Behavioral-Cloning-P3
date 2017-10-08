import csv
import cv2
import numpy as np
import sklearn
from random import shuffle

# read paths
def read_paths(paths):
    imgs = []
    for path in paths:
        with open(path) as csvf:
            readfile=csv.reader(csvf)
            for line in readfile:
                imgs.append(line)
    return imgs

#load dataset
data=['data/driving_log.csv', 'data2/driving_log.csv',
      'data_offroad_correct/driving_log.csv', 'data_shadow_track2/driving_log.csv',
      'data_track2/driving_log.csv', 'data_track2_2/driving_log.csv', 'data_curve/driving_log.csv']

data = read_paths(data)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample[0])
                center_angle = float(batch_sample[3])
                #flip the images, doubles the size of batch
                center_image = np.fliplr(cv2.imread(batch_sample[0]))
                center_angle = -float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Lambda, Cropping2D
from keras.layers.core import Dropout
from keras.layers.advanced_activations import PReLU

from sklearn.model_selection import train_test_split
train_samples, valid_samples = train_test_split(data, test_size=0.2)

rate = 0.25
rateconv = 0.25
batch_size = 16

train_generator=generator(train_samples, batch_size=batch_size)
valid_generator=generator(valid_samples, batch_size=batch_size)

nn = Sequential()
#crop top of image
nn.add(Cropping2D(cropping=((50,0), (0,0)), input_shape=(160,320,3)))
#normalize input
nn.add(Lambda(lambda x : x/126-1))
#fully convolutional network
nn.add(Conv2D(24, (15, 15), strides=(3,6), activation = 'relu'))
nn.add(Dropout(rateconv))
nn.add(Conv2D(48, (7,7), strides=(3,3), activation='relu'))
nn.add(Dropout(rateconv))
nn.add(Conv2D(96, (5,5), activation='relu'))
nn.add(Dropout(rateconv))
nn.add(Conv2D(192, (3,3), activation='relu'))
nn.add(Dropout(rateconv))
nn.add(Flatten())
nn.add(Dense(1))

nn.compile(loss='mse', optimizer='adam')
nn.fit_generator(train_generator, steps_per_epoch=
            len(train_samples)/batch_size+ len(train_samples)%batch_size, validation_data=valid_generator,
            validation_steps=len(valid_samples)/batch_size + len(valid_samples)%batch_size, epochs=60)

nn.save('model.h5')
