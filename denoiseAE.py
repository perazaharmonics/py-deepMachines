import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

# Auto-Encoding Denoising Series Filter-Bank

# We will train the Network using the original MNIST digits with shape (samples, 3, 28, 28) and normalize pixel values to be between 0-1
(X_train, _), (X_test, _) = mnist.load_data()

# Normalizing vectors to 0-1 dividing by color-level dynamic range
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))

noise_factor = 0.63
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n+1):
    ax = plt.subplot(1, n, i)
    plt.imshow(X_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# In order to properly denoise this Conv Autoencoder, we will need to a more robust Filter Bank.
# We will use the following filter bank:
# 1. 3x3 Convolution with 32 filters
# 2. 2x2 Max Pooling
# 3. 3x3 Convolution with 32 filters
# 4. 2x2 Max Pooling

# Input layer
input_img = keras.Input(shape=(28, 28, 1))
# 1st Lattice of Convolution and Max Pooling
encoder = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
encoder = layers.MaxPooling2D((2, 2), padding='same')(encoder)
# 2nd Lattice of Convolution and Max Pooling
encoder = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder)
conv_ae_sig = layers.MaxPooling2D((2, 2), padding='same')(encoder)

# At this point the representation is (7, 7, 32)

# 1st Lattice of Convolution and UpSampling 
decoder = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv_ae_sig)
decoder = layers.UpSampling2D((2, 2))(decoder)
# 2nd Lattice of Convolution and UpSampling
decoder = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(decoder)
decoder = layers.UpSampling2D((2, 2))(decoder)
# Output Layer
inv_conv_ae_sig = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoder)

denoise_ae = keras.Model(input_img, inv_conv_ae_sig)
denoise_ae.compile(optimizer='adam', loss='binary_crossentropy')

# Use TensorBoard to visualize the training process
tensorboard = TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=False)
model_file = "model_denoise_ae.h5"
checkpoint = keras.callbacks.ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

denoise_ae.fit(X_train_noisy, X_train, epochs=50, batch_size=128, shuffle=True, validation_data=(X_test_noisy, X_test), verbose=1, callbacks=[tensorboard, checkpoint])

# Show the denoised images
decoded_images = denoise_ae.predict(X_test)

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
