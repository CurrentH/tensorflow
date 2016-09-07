import tensorflow as tf
import numpy as np

from functions import create_samples
from functions import plot_clusters
from functions import choose_random_centroids
from functions import assign_to_nearest
from functions import update_centroids 

n_features = 2
n_clusters = 3
n_samples_per_cluster = 500
seed = np.random.seed()
embiggen_factor = 70

data_centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
initial_centroids = choose_random_centroids(samples, n_clusters)
nearest_indices = assign_to_nearest( samples, initial_centroids )
updated_centroids = update_centroids( samples, nearest_indices, n_clusters )

model = tf.initialize_all_variables()
with tf.Session() as session:
	sample_values = session.run(samples)
	updated_centroid_value = session.run(initial_centroids)
	print( updated_centroid_value )

plot_clusters(sample_values, updated_centroid_value, n_samples_per_cluster)

