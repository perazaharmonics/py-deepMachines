import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import CNN1
import matplotlib.pyplot as plt

def load_and_preprocess_data():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Convert class vectors to binary class matrices (for use with categorical_crossentropy)
    y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

def main():
    # Load and preprocess the CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    # Define the model parameters
    batchSize = 64
    classNum = 10
    dropOut = 0.5
    learningRate = 0.001
    epochs = 10
    imageSize = (32, 32)

    # Define label dictionary
    label_dict = {
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

    # Create a Data object
    dataObj = CNN1.Data('path_to_data', 'filename', batchSize, imageSize, seed=42, learningRate=learningRate, dropOut=dropOut)

    # Create a CNNModel object
    model = CNN1.CNNModel(batchSize, imageSize, classNum, dropOut, learningRate, epochs)
    dataObj.images, dataObj.labels = x_train, y_train  # Assuming Data class can directly accept preloaded data

    # Train the model
    history = model.train(dataObj, (x_test, y_test))

    # Optionally, save the model
    model.save('path_to_save_model')

    # Plot the images
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_train[i])
        ax.axis('off')
        ax.set_title(f'Label: {label_dict[np.argmax(y_train[i])]}')
    plt.tight_layout()
    plt.show()

    # Plot accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Add other helpful plots here

if __name__ == '__main__':
    main()
