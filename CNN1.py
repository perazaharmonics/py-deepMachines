import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import sys

import tensorflow as tf
from sklearn.utils import shuffle



class Data:
    def __init__(self, dataDir, fileName, batchSize, seed, classNum=10):
        self.dataDir = dataDir
        self.fileName = fileName
        self.classNum = classNum
        self.batchSize = batchSize
        self.seed = seed

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

    def load_data_batch(self):
        with open(os.path.join(self.dataDir, self.fileName), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self.images = data['data']
            self.labels = data['labels']

    def reshape_data(self):
        self.images = self.images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        self.labels = np.array(self.labels)

    def visualize_data(self, indices):
        plt.figure(figsize=(5, 5))

        for i in range(len(indices)):
            pic = self.images[indices[i]]
            label = self.labels[indices[i]]

            plt.subplot(2, 2, i + 1)
            plt.imshow(pic)
            plt.title(self.labelsDicti[label])

        plt.show()

# Next we make an object of the Data class and call the functions we have built on the object
# to visualize it in the following manner:

# Get the image data
DATA_DIR = 'cifar-10-batches-py'

#Hyperparameters for the model

BATCH_SIZE = 128
CLASS_NUM = 10
EPOCHS = 20
DROPOUT = 0.5
LEARNING_RATE = 0.001
SEED = 2

dataObj = Data(DATA_DIR, 'data_batch_1', BATCH_SIZE, SEED)
dataObj.load_data_batch()
dataObj.reshape_data()
dataObj.visualize_data([100, 4000, 2, 8000])

# here we have chosen indices 100, 400, 2, 8000.

# Use one-hot encoding to convert the labels to binary categorical vectors
# of where the length of the vector is equal to the number of unique categories

def  one_hot_encoding(self):

    # this function will conver the labels into one-hot encoding vectors
    # to facilitate data processing
    # initially the label vector is a list, we will conver it to a numpy array

    self.labels = np.array(self.labels, dtype=np.int32)

    # converting to one-hot
    self.labels = np.eye(self.classNum)[self.labels]

    print(self.labels.shape)

def normalize_images(self):

    self.images = self.images / (255.0 - 0000.5) # divide by 255 color levels minus epsilon to normalize the images and avoid dividing by zero


# To facilitate proper training, we need to bring up random samples. 
# We will use the shuffle function from the sklearn library to shuffle the data.
def shuffle_data(self):

    #shuffle the data so that training is more effective
    self.images, self.labels = shuffle(self.images, self.labels, random_state=self.seed)

# The next functions holds importance for the data class. The function will generate batches of data and labels
# from the loaded file. We know that we train our model in batches and we have declared a hyper-parameter for the batch size.

def generate_batches(self):

    # function to yield out batches of batchSize from the loaded file
    for i in range(0, len(self.images, self.batchSize)):

        end = min(i + self.batchSize, len(self.images))

        yield (self.images[i: end], self.labels[i: end])

# Now we will define our CNN model using TensorFlow. We will define the model in a class called CNNModel.   
        
class CNNModel:
    def __init__(self, batchSize, classNum, dropOut, learningRate, epochs, imageSize, savePath):
        self.batchSize = batchSize
        self.classNum = classNum
        self.dropOut = dropOut
        self.learningRate = learningRate
        self.epochs = epochs
        self.imageSize = imageSize
        self.savePath = savePath

        with tf.name_scope('placeholders') as scope:

            self.x = tf.placeholder(shape = [None, self.imageSize[0],
                                             self.imagSize[1], 3], dtype = tf.float32, name = 'inp_x')
            self.y = tf.placeholder(shape = [None, self.classNum], dtype = tf.float32, name = 'true_y')
            self.keepProb = tf.placeholder(dtype = tf.float32)

# Tuning the network architecture involes using different filter numbers, kernel sizes, 
# and a varying number of layers in the network. Let's define the first layer of our model. 
# We will use the 64 filters in the first layer with a kernel size of 3x3:
            
            with tf.name_scope('conv_1') as scope:
                # tensorflow takes the kernel as a 4D tensor. We can initialize the values the values with tf.zeros

                filter1 = tf.Variable(tf.zeros([3, 3, 3, 64], dtype=tf.float32),
                                      name='filter_1')
                conv1 = tf.nn.relu(tf.nn.conv2d(self.x, filter1, [1, 1, 1, 1], padding='SAME', name = 'convo_1'))

# In tensorflow, we need to define the filters as a variable 4D tensor. The first three
# dimensions represent the filter rows and cols (2D Filter) and the fourth dimensions is the 
# number of filters we want. Here, the third dimension had to be the current depth and the fourth dimensions has to be 
# the number of filters we want.
                
# We will use a max-pooling layer to reduce the spatial dimensions of the input image.
                with tf.name_scope('pool_1') as scope:
                    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxPool_1')

# The next layer will have 128 filters with a kernel size of 3x3. We will use the same max-pooling layer to reduce the spatial dimensions of the input image.
                with tf.name_scope('conv_2') as scope:
                    filter2 = tf.Variable(tf.zeros([2, 2, 64, 128], dtype=tf.float32), name='filter_2')
                    conv2 = tf.nn.relu(tf.nn.conv2d(pool1, filter2, [1, 1, 1, 1], padding='SAME', name='convo_2'))
                with tf.name_scope('conv_3') as scope:
                    filter3 = tf.Variable(tf.zeros([2, 2, 128, 128], dtype=tf.float32), name='filter_3')
                    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, filter3, [1, 1, 1, 1], padding='SAME', name='convo_3'))
                with tf.name_scope('pool_2') as scope:
                    
                    pool2 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxPool_2')
                    

                with tf.name_scope('conv_4') as scope:
                    filter4 = tf.Variable(tf.zeros([1, 1, 128, 256], dtype=tf.float32), name='filter_4')
                    conv4 = tf.nn.relu(tf.nn.conv2d(pool2, filter4, [1, 1, 1, 1]))

                    conv4 = tf.nn.relu(tf.nn.conv2d(conv3, filter4, [1, 1, 1, 1], padding='SAME', name='convo_4'))

                with tf.name_scope('pool_3') as scope:

                    pool3 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxPool_3')

                with tf.name_scope('conv_5') as scope:
                    filter5 = tf.Variable(tf.zeros([1, 1, 256, 512], dtype=tf.float32), name='filter_5')
                    conv5 = tf.nn.relu(tf.nn.conv2d(pool3, filter5, [1, 1, 1, 1], padding='SAME', name='convo_5'))

# Now it's time to add the fully connected layers. To add the fully connected layers, we will first need to flatten the output
# 
                    with tf.name_scope('flatten') as scope:
                        flat = tf.contrib.layers.flatten(conv5)

