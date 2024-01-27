import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras.layers import Dense
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix

import matplotlib.pyplot as plt
data = pd.read_csv("creditcard.csv").drop(['Time'], axis=1)
print(data.shape)

print('Number of fraudulent transactions = ', sum(data.Class == 1))

print('Number of valid transactions = ', sum(data.Class == 0))

data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

'After preprocessing the data, we split the data into training and test sets'

np.random.seed(1)
X = data.drop(['Class'], axis=1).values

'Training the model on normal cases and testing on fraud cases'

X_train, X_test, y_train, y_test = train_test_split(X, data['Class'], test_size=0.2, stratify=data['Class'])

# The input layer requires the special input_shape parameter which should match
input_size = 29
input_layer = Input(shape=(input_size,))

# The second layer is the hidden layer with 40 neurons
hidden_size = 40
encoder = Dense(hidden_size, activation="relu")(input_layer)

# The third layer is the output layer with 29 neurons (same as the input layer)
decoder = Dense(input_size, activation="sigmoid")(encoder)

# Connect the layers together
ae = Model(inputs=input_layer, outputs=decoder)
print(ae.summary())

# Compile the model with Adam optimizer, learning rate of 0.001 and mean squared error as loss function
ae.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')

# Use TensorBoard to visualize the training process
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=False)
model_file = "model_ae.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Fit the model with data (X_train, X_train) and validate with data (X_test, X_test) as autoencoders
num_epoch = 30
batch_size = 64

# Check the shapes right before fitting
print(X_train.shape)  # This should be (number of samples, 29)
print(X_test.shape)   # This should also be (number of samples, 29)

ae.fit(X_train, X_train, epochs=num_epoch, batch_size=batch_size, shuffle=True, validation_data=(X_test, X_test), verbose=1, callbacks=[tensorboard, checkpoint])


# Now we feed the testing set to the well-trained model and compute the reconstruction error that's measured by mse

recon = ae.predict(X_test)
recon_error = np.mean(np.square(X_test - recon, 2), axis = 1)

# evalute the area under the ROC curve to evaluate the binary classification problem on imbalanced data
roc_auc = roc_auc_score(y_test, recon_error)
print('Area Under ROC Curve = ', roc_auc)

# The Precision-Recal curve is a better way to detect if the model performs well with small anomalies (fraud cases)
precision, recall, th = precision_recall_curve(y_test, recon_error)
plt.plot(recall, precision, 'b', label='Precision-Recall curve')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')

threshold = 0.000001
y_pred = [1 if e > threshold else 0 for e in recon_error]
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
plt.ylabel('Precision')
plt.show()
