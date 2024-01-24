import tensorflow as tf
import tensorflow_probability as tfp
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
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size

    def _init_parameters(self, id):
        W = self.add_weight(shape=(self.num_v, self.num_h), initializer='random_normal', trainable=True, name='W'+str(id))
        a = self.add_weight(shape=(self.num_v,), initializer='zeros', trainable=True, name='a'+str(id))
        b = self.add_weight(shape=(self.num_h,), initializer='zeros', trainable=True, name='b'+str(id))
        return W, a, b
    
    # Define a call method for the RBM using a sigmoid activation function
    def call(self, v):
        # Perform Gibbs sampling and return the final visible layer
        _, _, vk, _ = self._gibbs_sampling(v)
        return tf.sigmoid(vk)

    def _gibbs_sampling(self, v):
        print("Shape of visible units v:", v.shape)
        print("Shape of weight matrix W:", self.W.shape)
        print("Shape of visible bias a:", self.a.shape)
        print("Shape of hidden bias b:", self.b.shape)
        v0 = v
        prob_h_v0 = self._prob_h_given_v(v0)
        vk = v
        prob_h_vk = prob_h_v0
        for _ in range(self.k):
            hk = self._bernoulli_sampling(prob_h_vk)
            prob_v_hk = self._prob_h_given_v(vk)
            vk = self._bernoulli_sampling(prob_v_hk)
            prob_h_vk = self._prob_h_given_v(vk)

        mask = tf.cast(tf.equal(v0, 0), dtype=tf.float32)
        vk = mask * v0 + (1 - mask) * vk
        prob_h_vk = prob_h_vk * mask + prob_h_v0 * (1 - mask)
        print("v0 shape:", v0.shape)
        print("prob_h_v0 shape:", prob_h_v0.shape)
        print("prob_h_vk shape:", prob_h_vk.shape)
        return v0, prob_h_v0, vk, prob_h_vk

    def _prob_v_given_h(self, h):
        return tf.sigmoid(tf.add(self.a(h), tf.matmul(h, tf.transpose(self.W))))

    def _prob_h_given_v(self, v):
        return tf.sigmoid(tf.add(self.b(v), tf.matmul(v, self.W)))

    def _bernoulli_sampling(self, prob):
        distribution = tfp.distributions.Bernoulli(probs=prob, dtype=tf.float32)
        return tf.cast(distribution.sample(), tf.float32)

    def _compute_gradient(self, v0, prob_h_v0, vk, prob_h_vk):
        outer_product0 = tf.matmul(tf.transpose(v0), prob_h_v0)
        outer_productk = tf.matmul(tf.transpose(vk), prob_h_vk)
        W_grad = tf.reduce_mean(outer_product0 - outer_productk, axis=0)
        a_grad = tf.reduce_mean(v0 - vk, axis=0)
        b_grad = tf.reduce_mean(prob_h_v0 - prob_h_vk, axis=0)
        return W_grad, a_grad, b_grad

    def _optimize(self, v):
        v0, prob_h_v0, vk, prob_h_vk = self._gibbs_sampling(v)
        W_grad, a_grad, b_grad = self._compute_gradient(v0, prob_h_v0, vk, prob_h_vk)
        self.W.assign_add(self.learning_rate * W_grad)
        self.a.assign_add(self.learning_rate * a_grad)
        self.b.assign_add(self.learning_rate * b_grad)
        error = tf.reduce_mean(tf.square(v0 - vk))
        return error
    def train(self, X_train):
        """
        Model training
        @param X_train: training dataset
        """
        model = RBM(self.num_v, self.num_h, self.batch_size, self.learning_rate, self.num_epoch, self.k)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss=self.compute_loss)

        model.fit(X_train, X_train, batch_size=self.batch_size, epochs=self.num_epoch, verbose=1)

        # Print training error
        train_loss = model.evaluate(X_train, X_train, verbose=0)
        print("Training Error: ", train_loss)

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

    def __init__(self, layer_sizes, batch_size, learning_rates, num_epoch, k=2):
        super(DBN, self).__init__()
        self.rbms = []
        for i in range(1, len(layer_sizes)):
            rbm = RBM(num_v=layer_sizes[i-1], id=i,
                      num_h=layer_sizes[i], batch_size=batch_size,
                      learning_rate=learning_rates[i-1], num_epoch=num_epoch, 
                      k=k)
            self.rbms.append(rbm)

    def train(self, X_train):
        """
        Model training
        @param X_train: training dataset
        """
        self.rbms_para = []
        input_data = None
        for rbm in self.rbms:
            if input_data is None:
                input_data = X_train.copy()
            rbm.compile(optimizer=keras.optimizers.Adam(learning_rate=rbm.learning_rate),
                        loss=rbm.compute_loss)
            rbm.fit(input_data, input_data, batch_size=rbm.batch_size, epochs=rbm.num_epoch, verbose=1)
            parameters = (rbm.W.numpy(), rbm.a.numpy(), rbm.b.numpy())
            self.rbms_para.append(parameters)
            input_data = rbm.hidden_layer(input_data, parameters)

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
