from sklearn import load_digits
from matplotlib import pyplot as plt

digits = load_digits()

fig = plt.figure( figsize=(3,3) )

plt.imshow(digits['images'][66], cmap="gray", interpolation='none')

plt.show()


