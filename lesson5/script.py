import tensorflow as tf
import resource

print("{} Kb".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

session = tf.InteractiveSession()

x = tf.constant(list(range(10)))

print(x.eval())

session.close()
