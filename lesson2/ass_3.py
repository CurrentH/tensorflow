import tensorflow as tf
import numpy as np

for i in range(5):
	data = np.average(np.random.randint(1000, size=10000))

	temp = tf.Variable( data , name='temp')
	x = tf.Variable( 0 , name='x' )

	with tf.Session() as session:
		merged = tf.merge_all_summaries()
		writer = tf.train.SummaryWriter("./", session.graph)
		model = tf.initialize_all_variables()
		session.run(model)

		x = temp

		print(session.run(x))

