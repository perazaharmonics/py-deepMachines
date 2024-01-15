import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from tensorflow import keras

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Make a fashion object that'll contain the clothing/fashion data
fashion = keras.datasets.fashion_mnist

# trainX contains input images and trainY contains correspongin output labels\
# testX contains input images and testY contains correspongin output labels
(trainX, trainY), (testX, testY) = fashion.load_data()
color_scale = 256 # grayscale image of have 0-255 levels of color
print('train data x shape: ', trainX.shape)
print('test data x shape: ', testX.shape)

print('train data y shape: ', trainY.shape)
print('test data y shape: ', testY.shape)


# make a label dictionary to map integer labels to classes
classesDisct = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress', 
    4: 'Coat', 
    5: 'Sandal', 
    6: 'Shirt', 
    7: 'Sneaker', 
    8: 'Bag', 
    9: 'Ankle boot'

}

# Visualize the data
rows = 2
columns = 2
fig = plt.figure(figsize = (5,5))

for i in range(1, rows*columns + 1):
    image = trainX[i]
    label = trainY[i]

    sub = fig.add_subplot(rows, columns, i)
    sub.set_title('Label:' + classesDisct[label])

    plt.imshow(image)
plt.show()

# Normalize the data
# Normalize the data
trainX =trainX / (color_scale-1) # 255 (256-1) pixels for grayscale image 
testX = testX / (color_scale-1) # 255 (256-1) pixels for grayscale image 

trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.1, random_state=1)

# random_state is used for randomly shuffling the dataset

CLASS_NUM = 10
#number of clasees we need to classify

INPUT_UNITS = 28*28 #one neuron for each pixel in the image
#no. of neurons in input layer layer 784 as we have 28*28 pixels in the image

HIDDEN_LAYER1_UNITS = color_scale # 256 neurons in the first hidden layer

HIDDEN_LAYER2_UNITS = int(0.5*color_scale) # half of the color scale in the second hidden layer

OUTPUT_UNITS = CLASS_NUM # 10 neurons in the output layer
# no. of neurons in output layer = no. of classis to classify.
# each neuron will output the probability of the input image belonging to that class

# Adam optimizer parameters
learning_rate = 1e-3 #@param {type: 'number'} # You can adjust this value

BATCH_SIZE = 64
#we will take input data in sets of 64 images at once instead of using whole data for every iteration. 
# Each set is called a batch and a batch function is used to generate batches of data.

NUM_BATCHES = int(trainX.shape[0] / BATCH_SIZE)
# number of mini-batches required to cover the train data

EPOCHS = 10
# number of iterations we will perform to train train

# Create a one-hot encoder transformation matrix to convert integer labels to one-hot vectors

trainY = np.eye(CLASS_NUM)[trainY]
valY = np.eye(CLASS_NUM)[valY]
testY = np.eye(CLASS_NUM)[testY]


# Create a tensorflow graph (the tensorflow graphs is just a state-less filter for now)

with tf.name_scope('placeholders') as scope:

    # making placeholders for inputs (x) and outputs (y)
    # the first dimension 'BATCH_SIZE' represents the number of samples
    # in a batch. It can also be kept 'None'. Tensorflow will automatically
    # detect the shape from incoming data. 

    x = tf.placeholder(shape = [BATCH_SIZE, 784], dtype = tf.float32, name = 'input_x')
    y = tf.placeholder(shape = [BATCH_SIZE, CLASS_NUM], dtype = tf.float32, name = 'true_y')


# Add the layers and biases to the graph (filter)
    
with tf.name_scope('inp_layer') as scope:

    # the first set of weight will be connecting the input layers to the first hidden layer
    # Hence, it will be essentially a matrix of weights of shape [INPUT_UNITS, HIDDEN_LAYER1_UNITS]

    weights1 = tf.get_variable(shape=[INPUT_UNITS, HIDDEN_LAYER1_UNITS], dtype=tf.float32, name='weights_1')
    biases1 = tf.get_variable(shape=[HIDDEN_LAYER1_UNITS], dtype=tf.float32, name='biases_1')

    # performing W.x +b, we rather multiply x to W and multiply it to x due to Lin Alg constratints
    # otherwise you can also take transpose of W and multiply it to x

    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights1), biases1), name='layer_1')

    # we use the relu activation in the 2 hidden layers to introduce non-linearity in the model

with tf.name_scope('hidden_layer_1') as scope:
    
    # second set of weights between hidden layer 1 and hidden layer 2 
    weights_2 = tf.get_variable(shape=[HIDDEN_LAYER1_UNITS, HIDDEN_LAYER2_UNITS], dtype=tf.float32, name='weights_2')
    biases_2 = tf.get_variable(shape=[HIDDEN_LAYER2_UNITS], dtype=tf.float32, name='biases_2')


    # the output of layer 1 will be fed to layer 2 (as this is Feedforward Network)
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights_2), biases_2), name = 'layer_2')

with tf.name_scope('hidden_layer_2') as scope:
    
        # third set of weights between hidden layer 2 and output layer
        weights_3 = tf.get_variable(shape = [HIDDEN_LAYER2_UNITS, OUTPUT_UNITS], dtype = tf.float32, name = 'weights_3')
        biases_3 = tf.get_variable(shape = [OUTPUT_UNITS], dtype = tf.float32, name = 'biases_3')
    
        # the output of layer 2 will be fed to layer 3 (as this is Feedforward Network)
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights_3), biases_3), name = 'out_layer')

# Adding the loss function to the graph (filter)
# As this is a classification task, we will use the softmax cross entropy loss function
# The softmax function will give us the probability of the input image belonging to each class
# The cross entropy loss function will compute the loss by comparing the predicted probability
# distribution to the true probability distribution

with tf.name_scope('loss') as scope:
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = layer_3), name = 'loss')

# Adding the optimizer to the graph (filter)
# We will use the Adam optimizer to minimize the loss function
# Adam optimizer is an extension of the stochastic gradient descent optimizer
# It uses adaptive learning rates and momentum to converge faster

with tf.name_scope('optimizer') as scope:
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

    
# Adding the accuracy metric to the graph (filter)
# We will use the accuracy metric to evaluate the performance of the model
# The accuracy metric will compute the average accuracy over the batches
# The accuracy is the ratio of number of correctly classified images to the total number of images

with tf.name_scope('accuracy') as scope:
    # here we will check how many predictions how many predictions our model is making correct as compared to the labels

    correct_preds = tf.equal(tf.argmax(layer_3, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32), name = 'accuracy')

with tf.Session() as sess:
     
     sess.run(tf.global_variables_initializer())

     for epoch in range(EPOCHS):
         for batch in range(NUM_BATCHES):
             # we are using the trainX and trainY data to train the model
              
             # creating a batch of inputs
            batchX = trainX[i*BATCH_SIZE : (i+1)*BATCH_SIZE, :].reshape(-1, 784) # reshape to flatten image into tensor
            batchY = trainY[i*BATCH_SIZE : (i+1)*BATCH_SIZE, :]
             # running the train operation for updating weigths after very mini-batch
            _, miniBatchLoss, acc = sess.run([optimizer, loss, accuracy], feed_dict = {x: batchX, y: batchY})

            # printing the accuracy and loss  for 4th training batch
            if i % 10 == 0:
                print('Epoch: {}, Batch: {}, Loss: {}, Accuracy: {}'.format(epoch, batch, miniBatchLoss, acc))

                # caLculating the accuracy and loss for the validation data
                for i in range(int(valX.shape[0] / BATCH_SIZE)):

                    valBatchX = valX[i*BATCH_SIZE : (i+1)*BATCH_SIZE, :]
                    valBatchY = valY[i*BATCH_SIZE : (i+1)*BATCH_SIZE, :]
                    valLoss, valAcc = sess.run([loss, accuracy], feed_dict = {x: valBatchX, y: valBatchY})

                    if i % 5 == 0:
                        print('Validation Batch: ', i , 'Val loss: ', valLoss, 'val Acc: ', valAcc )

                # after training the model, testing performance on the test batch
                for i in range(int(testX.shape[0] / BATCH_SIZE)):

                    testBatchX = testX[i*BATCH_SIZE : (i+1)*BATCH_SIZE, :]
                    testBatchY = testY[i*BATCH_SIZE : (i+1)*BATCH_SIZE, :]
                    testLoss, testAcc = sess.run([loss, accuracy], feed_dict = {x: testBatchX, y: testBatchY})

                    if i % 5 == 0:
                        print('Test Batch: ', i , 'Test loss: ', testLoss, 'Test Acc: ', testAcc )
