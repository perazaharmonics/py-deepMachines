import numpy as np
import tensorflow as tf
from tensorflow import keras
from RBM import RBM

data_path = './ratings.dat'
num_users = 6040
num_movies = 3706
data = np.zeros([num_users, num_movies], dtype=np.float32)
movie_dict = {}

with open(data_path, 'r') as file:
    for line in file.readlines()[1:]:
        user_id, movie_id, rating, _ = line.split('::')
        user_id = int(user_id) - 1  # Convert user_id to zero-based index

        if movie_id not in movie_dict:
            movie_dict[movie_id] = len(movie_dict)

        # Convert rating to a float and scale between 0 and 1
        rating = float(rating) / 5
        data[user_id, movie_dict[movie_id]] = rating

# After finishing the loop, reshape the data if needed and print its shape
data = np.reshape(data, [data.shape[0], -1])
print(data.shape)  # This should output: (6040, 3706)

# The training dataset is of the size 6,040 x 3,706 with each row consisting of 3706 scaled ratings
# including 0.0, meaning not rated. 

# Take a look at their distribution
values, counts = np.unique(data, return_counts=True)
for value, count in zip(values,counts):
    print('Number of {:2.1f} ratings: {}'.format(value, count))

# Split the data into train and test sets
num_train = int(data.shape[0] * 0.8)
np.random.seed(1)
np.random.shuffle(data)
data_train, data_test = data[:num_train, :], data[num_train:, :]

# Create a mask for the test set to simulate missing ratings
sim_index = np.zeros_like(data_test, dtype=bool)
perc_sim = 0.2
for i, user_test in enumerate(data_test):
    exist_index = np.where(user_test > 0.0)[0]
    sim_index[i, np.random.choice(exist_index, int(len(exist_index)*perc_sim))] = True
data_test_sim = np.copy(data_test)
data_test_sim[sim_index] = 0.0

# Define the RBM model using Keras
model = keras.Sequential([
    keras.layers.Dense(80, activation='sigmoid', input_shape=(num_movies,)),
    keras.layers.Dense(num_movies, activation='sigmoid')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')

# Train the RBM model
model.fit(data_train, data_train, batch_size=64, epochs=100)

# Make predictions on the test set
prediction = model.predict(data_test_sim)

np.random.seed(0)
np.random.shuffle(data)
data_train, data_test = data[:num_train, :], data[num_train:,  :]
sim_index = np.zeros_like(data_test, dtype=bool)
perc_sim = 0.2
for i, user_test in enumerate(data_test):
    exist_index = np.where(user_test > 0.0)[0]
    sim_index[i, 
              np.random.choice(exist_index, int(len(exist_index)*perc_sim))] = True
data_test_sim = np.copy(data_test)
data_test_sim[sim_index] = 0.0

rbm = RBM(num_v=num_movies, num_h=80, batch_size=64, num_epoch=100, learning_rate=1, k=5)

prediction = rbm.predict(data_test_sim)

from sklearn.metrics import mean_squared_error
print("Test Error:", mean_squared_error(data_test[sim_index], prediction[sim_index]))
