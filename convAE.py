import keras
import tensorflow as tf
import numpy as np
from keras import layers
from keras.datasets import mnist
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

# Auto-Encoding Series "Filter-Bank"

# Input layer


input_img = keras.Input(shape=(28, 28, 1))

# Hidden Network 1 (Encoder) - Low Pass Filtering Layer
encoder = layers.Conv2D(16(3, 3), activation='relu', padding = 'same')(input_img)
encoder = layers.MaxPooling2D((2, 2), padding = 'same' )(encoder)
encoder = layers.Conv2d(8, (3, 3), activation='relu', padding='same')(encoder)
encoder = layers.MaxPooling2D((2, 2), padding = 'same' )(encoder)
encoder = layers.Conv2d(8, (3, 3), activation='relu', padding='same')(encoder)

# Low-Pass Filter Ouput

conv_ae_sig = layers.MaxPooling2D((2, 2), padding='same')(encoder)

# at this point in the network the representation is 128-dimensional i.e. (4, 4, 8)

# High - Pass Filtering Layer
decoder=layers.Conv2D(8, (3, 3), activation = 'relu', padding='same')(conv_ae_sig)
decoder=layers.UpSampling2D((2, 2))(decoder)
decoder=layers.Conv2D(16, (3, 3), activation='relu')(decoder)
decoder=layers.UpSampling2D((2, 2))(decoder)
decoder=layers.Conv2D(16, (3, 3), activation='relu')(decoder)
decoder=layers.UpSampling2D((2, 2))(decoder)

inv_conv_ae_sig = layers.Conv2D(1, (3, 3), activation = 'sigmoid', padding='same')(decoder)

conv_ae = keras.Model(input_img, inv_conv_ae_sig)
conv_ae.compile(optimizer='adam', loss='binary_crossentropy')

# We will train the Network using the original MNIST digits with shape (samples, 3, 28, 28) and normalize pixel values to be between 0-1
(X_train, _), (X_test, _) = mnist.load_data()

# Normalizing vectors to 0-1 dividing by color-level dynamic range
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))

# Use TensorBoard to visualize the training process
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=False)
model_file = "model_l1sparse_ae.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


conv_ae.fit(X_train, X_train, epochs=50, batch_size=128, shuffle=True, validation_data=(X_test, X_test), verbose=1, callbacks=[tensorboard, checkpoint])

decoded_images = conv_ae.predict(X_test)

n = 10
plt.figure(figsize=(20, 4))

for i in range(1, n+1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Display reconstruction
    ax = plt.subplot(2, n, i+n)
    plt.imshow(decoded_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

encoded = keras.Model(input_img, conv_ae_sig)
encoded_imgs = encoder.predict(X_test)

n = 10
plt.figure(figsize=(20, 8))
for i in range(1, n+1):
    ax = plt.subplot(1, n, i)
    plt.imshow(encoded_imgs[i].reshape(4, 4*8).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
