import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp


   
class RBM(keras.Model):
    def __init__(self, num_v, num_h, batch_size, learning_rate, num_epoch, k=2):
        super(RBM, self).__init__()
        self.num_v = num_v
        self.num_h = num_h
        self.k = k
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch

        self.W = self.add_weight(shape=(num_v, num_h), initializer='random_normal', trainable=True)
        self.a = self.add_weight(shape=(num_v,), initializer='zeros', trainable=True)
        self.b = self.add_weight(shape=(num_h,), initializer='zeros', trainable=True)



    def call(self, v):
        # Perform Gibbs sampling and return the final visible layer
        _, _, vk, _ = self._gibbs_sampling(v)
        return vk

    def _gibbs_sampling(self, v):
        print("Shape of visible units v:", v.shape)
        print("Shape of weight matrix W:", self.W.weights[0].shape)
        print("Shape of visible bias a:", self.a.weights[0].shape)
        print("Shape of hidden bias b:", self.b.weights[0].shape)
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
        return tf.sigmoid(tf.add(self.a(h), tf.matmul(h, tf.transpose(self.W.weights[0]))))

    def _prob_h_given_v(self, v):
        return tf.sigmoid(tf.add(self.b(v), tf.matmul(v, self.W.weights[0])))

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
        self.W.weights[0].assign_add(self.learning_rate * W_grad)
        self.a.weights[0].assign_add(self.learning_rate * a_grad)
        self.b.weights[0].assign_add(self.learning_rate * b_grad)
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

    def compute_loss(self, v0, vk):
        # Compute the loss function (e.g., mean squared error)
        return tf.reduce_mean(tf.square(v0 - vk))

    def predict(self, v):
        prob_h_v = self.call(v)
        h = tf.cast(tf.random.uniform(tf.shape(prob_h_v)) < prob_h_v, tf.float32)
        prob_v_h = self.call(h)
        return prob_v_h

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

        mask = tf.cast(tf.equal(v0,0), dtype=tf.float32)
        vk = mask * v0 + (1 - mask) * vk
        prob_h_vk = prob_h_vk * mask + prob_h_v0 * (1 - mask)
        print("v0 shape:", v0.shape)
        print("prob_h_v0 shape:", prob_h_v0.shape)
        print("prob_h_vk shape:", prob_h_vk.shape)
        return v0, prob_h_v0, vk, prob_h_vk

    def _prob_v_given_h(self, h):
        return tf.sigmoid(tf.add(self.a, tf.matmul(h, tf.transpose(self.W))))

    def _prob_h_given_v(self, v):
        return tf.sigmoid(tf.add(self.b, tf.matmul(v, self.W)))

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
        model = keras.Sequential()
        model.add(keras.layers.Dense(self.num_h, activation='sigmoid', input_shape=(self.num_v,)))
        model.add(keras.layers.Dense(self.num_v, activation='sigmoid'))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss=self.compute_loss)

        model.fit(X_train, X_train, batch_size=self.batch_size, epochs=self.num_epoch, verbose=1)

        # Print training error
        train_loss = model.evaluate(X_train, X_train, verbose=0)
        print("Training Error: ", train_loss)


    def compute_loss(self, v0, vk):
        # Compute the loss function (e.g., mean squared error)
        return tf.reduce_mean(tf.square(v0 - vk))


    def predict(self, v):
        prob_h_v = tf.sigmoid(tf.matmul(v, self.W) + self.b)
        h = self._bernoulli_sampling(prob_h_v)
        prob_v_h = tf.sigmoid(tf.matmul(h, tf.transpose(self.W)) + self.a)
        return prob_v_h
