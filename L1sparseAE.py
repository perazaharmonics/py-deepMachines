import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers

from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix 
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("creditcard.csv").drop(['Time'], axis=1)
print(data.shape)

print('Number of fraudulent transactions = ', sum(data.Class == 1))
print('Number of valid transactions = ', sum(data.Class == 0))

# Preprocess the data
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

# Split the data into training and test sets
np.random.seed(1)
X = data.drop(['Class'], axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, data['Class'], test_size=0.95, stratify=data['Class'])

# Build the Deep Autoencoder Network
input_size = 29
input_layer = Input(shape=(input_size,))
hidden_sizes = [80, 40, 80]
encoder = Dense(hidden_sizes[0], activation="relu", activity_regularizer=regularizers.l1(3e-5))(input_layer)
encoder = Dense(hidden_sizes[1], activation="relu")(encoder)
decoder = Dense(hidden_sizes[2], activation="relu")(encoder)
decoder = Dense(input_size)(decoder)
sparse_ae = Model(inputs=input_layer, outputs=decoder)
print(sparse_ae.summary())

optimizer = tf.keras.optimizers.Adam(lr=0.0008)
sparse_ae.compile(optimizer=optimizer, loss='mse')
# Use TensorBoard to visualize the training process
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=False)
model_file = "model_l1sparse_ae.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

num_epoch = 30
batch_size = 64
# Check the shapes right before fitting
print(X_train.shape)  # This should be (number of samples, 29)
print(X_test.shape)   # This should also be (number of samples, 29)

# Train the model
sparse_ae.fit(X_train, X_train, epochs=num_epoch, batch_size=batch_size, shuffle=True, validation_data=(X_test, X_test), verbose=1, callbacks=[tensorboard, checkpoint])

# Now we feed the testing set to the well-trained model and compute the reconstruction error that's measured by mse
recon = sparse_ae.predict(X_test)
recon_error = np.mean(np.square(X_test - recon), axis=1)

# Evaluate the area under the ROC curve to evaluate the binary classification problem on imbalanced data
roc_auc = roc_auc_score(y_test, recon_error)
print('Area Under ROC Curve = ', roc_auc)

# The Precision-Recall curve is a better way to detect if the model performs well with small anomalies (fraud cases)
precision, recall, th = precision_recall_curve(y_test, recon_error)
plt.plot(recall, precision, 'b', label='Precision-Recall curve')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
pr_auc = auc(recall, precision)

print('Area under precision-recall curve:', pr_auc)
threshold = 0.000001
y_pred = [1 if e > threshold else 0 for e in recon_error]
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
