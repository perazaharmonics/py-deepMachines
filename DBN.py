import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


# Create initial RBM layer (class)
class RBM(keras.Model):
    def __init__(self, num_v, id, num_h, batch_size,
                 learning_rate, num_epoch, k=2):
        super(RBM, self).__init__()
        self.num_v = num_v
        self.num_h = num_h
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.k = k
        self.W, self.a, self.b = self._init_parameters(id)
        
    def _init_parameters(self, id):
        """
        Initializing parameters for the id-th model
        including weights and biases
        @param id: the id-th model
        """
        W = self.add_weight(shape=(self.num_v, self.num_h), initializer='random_normal', trainable=True, name='W'+str(id))
        a = self.add_weight(shape=(self.num_v,), initializer='zeros', trainable=True, name='a'+str(id))
        b = self.add_weight(shape=(self.num_h,), initializer='zeros', trainable=True, name='b'+str(id))
        return W, a, b
# Define RBM train function using contrastive divergence
    def train(self, input_data):
        """
        Training the model using contrastive divergence
        @param input_data: input data for training
        """
        for epoch in range(self.num_epoch):
            for batch in range(0, len(input_data), self.batch_size):
                batch_data = input_data[batch: batch + self.batch_size]
                v0 = batch_data
                for k in range(self.k):
                    _, prob_h_v0, vk, prob_h_vk = self._gibbs_sampling(v0)
                    v0 = vk
                self._update_parameters(batch_data, prob_h_v0, vk, prob_h_vk)
        return self.W, self.a, self.b
    
# Define hidden layer function
    def hidden_layer(self, input_data, parameters):
        """
        Computing the output of the hidden layer
        @param input_data: input data for training
        @param parameters: parameters of the model
        """
        W, a, b = parameters
        return tf.sigmoid(tf.add(b, tf.matmul(input_data, W)))


# Create Deep-Belief Network class
class DBN(keras.Model):
    def __init__(self, num_v, layers, batch_size, learning_rate, num_epoch, k=2):
        super(DBN, self).__init__()
        self.num_v = num_v
        self.layers = layers
        self.k = k
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch

        self.rbm_layers = []
        for num_h in layers:
            self.rbm_layers.append(self._create_rbm_layer(num_v, num_h))
            num_v = num_h

    def call(self, v):
        h = v
        for rbm in self.rbm_layers:
            h = rbm.hidden_layer(h)
        return h
    
# Create Stacked-RBM layers
    def _create_rbm_layer(self, num_v, num_h):
        rbm = RBM(num_v, num_h, self.batch_size, self.learning_rate, self.num_epoch, self.k)
        return rbm

# Define update parameters function
    def _update_parameters(self, input_data):
        """
        Updating parameters
        @param input_data: input data for training
        """
        with tf.GradientTape() as tape:
            v0, prob_h_v0, vk, prob_h_vk = self._gibbs_sampling(input_data)
            loss = self._compute_loss(v0, prob_h_v0, vk, prob_h_vk)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

# Define train function
    def train(self, X_train):
        """
        Model training
        @param X_train: input data for training
        """
        self.rbms_para = []
        for rbm in self.rbms:
            if input_data is None:
                input_data = X_train.copy()
            parameters = rbm.train(input_data)
            self.rbms_para.append(parameters)
            input_data = rbm.hidden_layer(input_data, parameters)
# Define predict function
    def predict(self, X):
        """
        Computing the output of the last layer
        @param X: input data for training
        """
        data = None
        for rbm, parameters in zip(self.rbms, self.rbms_para):
            if data is None:
                data = X.copy()
            data = rbm.hidden_layer(data, parameters)
        return data
    
