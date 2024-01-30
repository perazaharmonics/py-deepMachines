import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Layer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

class ContractiveLossLayer(Layer):
    def __init__(self, taps, **kwargs):
        self.is_placeholder = True
        self.taps = taps
        super(ContractiveLossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        x, h = inputs
        W = self.taps
        W_t = tf.transpose(W)
        W_t_pwrsum = tf.reduce_sum(W_t ** 2, axis=1)
        contract = w * tf.reduce_sum((h * (1 - h)) ** 2 * W_t_pwrsum, axis=1)
        contract = tf.reduce_mean(contract)
        self.add_loss(contract, inputs=x)
        return x

data = pd.read_csv("creditcard.csv").drop(['Time'], axis=1)
print(data.shape)

print('Number of fraudulent transactions = ', sum(data.Class == 1))
print('Number of valid transactions = ', sum(data.Class == 0))

data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
print(data.shape)
# After preprocessing the data, we split the data into training and test sets
np.random.seed(1)
X = data.drop(['Class'], axis=1).values

# Training the model on normal cases and testing on fraud cases
X_train, X_test, y_train, y_test = train_test_split(X, data['Class'], test_size=0.2, stratify=data['Class'])

# The input layer requires the special input_shape parameter which should match
input_size = 29
input_layer = Input(shape=(input_size,))

# The second layer is the hidden layer with 40 neurons
hidden_size = 40
w = 1e-5

# Define the encoder as a separate model
encoder_layer = Dense(hidden_size, activation='sigmoid')(input_layer)
encoder = Model(inputs=input_layer, outputs=encoder_layer)

# Define the decoder
decoder_layer = Dense(input_size)(encoder_layer)
decoder = Model(inputs=input_layer, outputs=decoder_layer)

# The autoencoder is the combination of the encoder and the decoder
contract_ae = Model(inputs=input_layer, outputs=decoder_layer)
print(contract_ae.summary())

# Add the contractive loss layer to the model
taps = contract_ae.layers[1].get_weights()[0] #weights of the hidden layer = taps of the filter
contract_ae = Model(inputs=input_layer, outputs=ContractiveLossLayer(taps)([decoder_layer, encoder_layer]))

# Compile the model
contract_ae.compile(optimizer=tf.keras.optimizers.Adam(0.0003), loss='mse')

# Use TensorBoard to visualize the training process
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=False)
model_file = "model_contract_ae.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Fit the model with data (X_train, X_train) and validate with data (X_test, X_test) as autoencoders
num_epoch = 30
batch_size = 64

# Check the shapes right before fitting
print(X_train.shape)  # This should be (number of samples, 29)

# Fix: Pass the correct arguments for X_train and X_test
contract_ae.fit(X_train, X_train, epochs=num_epoch, batch_size=batch_size, shuffle=True, validation_data=(X_test, X_test), verbose=1, callbacks=[tensorboard, checkpoint])
print(X_test.shape)   # This should also be (number of samples, 29)

contract_ae.fit(X_train, X_train, epochs=num_epoch, batch_size=batch_size, shuffle=True, validation_data=(X_test, X_test), verbose=1, callbacks=[tensorboard, checkpoint])

# Now we feed the testing set to the well-trained model and compute the reconstruction error that's measured by mse
recon = contract_ae.predict(X_test)
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

threshold = 0.000001
y_pred = [1 if e > threshold else 0 for e in recon_error]
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
loss_value, gradients = grad(contract_ae, X_train)
print(loss_value)
print(gradients)
