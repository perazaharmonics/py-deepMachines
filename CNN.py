import numpy as np
import pickle
import os
import tensorflow as tf
import tensorrt as trt
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

class Data:
    def __init__(self, dataDir, fileName, batchSize, imageSize, seed, learningRate, dropOut, classNum=10):
        self.dataDir = dataDir
        self.fileName = fileName
        self.classNum = classNum
        self.batchSize = batchSize
        self.imageSize = imageSize
        self.seed = seed
        self.learningRate = learningRate
        self.dropOut = dropOut
        self.model = self.build_model()
        self.model.compile(optimizer=Adam(learning_rate=self.learningRate),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.summary()

        self.labelsDicti = {
            0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'
        }
    # Rest of the code remains the same

    def load_data_batch(self):
        with open(os.path.join(self.dataDir, self.fileName), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self.images = data['data']
            self.labels = data['labels']

    def reshape_data(self):
        self.images = self.images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        self.labels = np.array(self.labels)

    def one_hot_encoding(self):
        self.labels = to_categorical(self.labels, self.classNum)

    def normalize_images(self):
        self.images = self.images / 255.0

    def shuffle_data(self):
        self.images, self.labels = shuffle(self.images, self.labels, random_state=self.seed)

    def generate_batches(self):
        for i in range(0, len(self.images), self.batchSize):
            end = min(i + self.batchSize, len(self.images))
            yield (self.images[i: end], self.labels[i: end])

    def augment_data(self):
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
        datagen.fit(self.images)

        augmented_images = []
        augmented_labels = []

        for batchX, batchY in self.generate_batches():
            augmented_batchX = datagen.flow(batchX, batch_size=self.batchSize, shuffle=False).next()
            augmented_images.append(augmented_batchX)
            augmented_labels.append(batchY)

        self.images = np.concatenate(augmented_images)
        self.labels = np.concatenate(augmented_labels)

    def build_model(self):
        model = Sequential([
            Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(self.imageSize[0], self.imageSize[1], 3)),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(256, (3, 3), padding='same', activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(512, (3, 3), padding='same', activation='relu'),
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(self.dropOut),
            Dense(512, activation='relu'),
            Dropout(self.dropOut),
            Dense(self.classNum, activation='softmax')
        ])
        return model

class CNNModel:
    def __init__(self, batchSize, imageSize, classNum, dropOut, learningRate, epochs):
        self.batchSize = batchSize
        self.imageSize = imageSize  # Make sure this is a tuple or list, not an integer
        self.classNum = classNum
        self.dropOut = dropOut
        self.learningRate = learningRate
        self.epochs = epochs
        self.model = self.build_model()

        self.model.compile(optimizer=Adam(learning_rate=self.learningRate),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, dataObj, val_data=None):
        history = self.model.fit(dataObj.images, dataObj.labels, validation_data=val_data, epochs=self.epochs, verbose=1, batch_size=self.batchSize)
        return history

    def save(self, savePath):
        self.model.save(savePath)

    def build_model(self):
        model = Sequential([
            Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(self.imageSize[0], self.imageSize[1], 3)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(256, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(512, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(self.dropOut),
            Dense(512, activation='relu'),
            Dropout(self.dropOut),
            Dense(self.classNum, activation='softmax')
        ])
        return model
