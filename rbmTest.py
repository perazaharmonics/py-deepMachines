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


rbm = RBM(num_v=num_movies, num_h=80, learning_rate=0.01, batch_size = 64, num_epoch = 100,  k=5)
parameters_trained = rbm.train(data)
prediction = rbm.predict(data)

sample, sample_pred = data[0], prediction[0]
five_star_index = np.where(sample == 1.0)[0]
high_index = np.where(sample_pred >= 0.9)[0]
index_movie = {value: key for key, value in movie_dict.items()}
print('Movies with 5-star ratings:', ', '.join(index_movie[index] for index in five_star_index))
