import pickle
import gzip
import wget
import numpy as np
import matplotlib.pyplot as plt

from multisvm import Model, Trainer

# Load the dataset
try:
    f = gzip.open('mnist_py3k.pkl.gz', 'rb')
except:
    print("Could not find MNIST, downloading the dataset")
    wget.download("http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist_py3k.pkl.gz")
    f = gzip.open('mnist_py3k.pkl.gz', 'rb')
(xtr, ytr), (xval, yval), (xte, yte) = pickle.load(f)
# Need to convert to keras format
f.close()

xtr = xtr.reshape((-1, 1,  28, 28))  # Should be (Channel Height, Width)
xval = xval.reshape((-1, 1,  28, 28))  # Should be (Channel Height, Width)

print(np.max(xtr))
print("Training Data Shape: ", xtr.shape)
print("Training Labels Shape: ", ytr.shape)
print("Validation Data Shape: ", xval.shape)
print("Validation Labels Shape: ", yval.shape)

# Visualize an image
ind = np.random.randint(xtr.shape[0])
plt.imshow(xtr[ind, 0, :, :], cmap='gray')
plt.title("Digit = %s" % ytr[ind])
plt.show()

def flatten(x):
    return x.reshape(x.shape[0], -1)

xtr = xtr.reshape((-1, 1,  28, 28))  # Should be (Channel Height, Width)
xval = xval.reshape((-1, 1,  28, 28))  # Should be (Channel Height, Width)

Model(xtr.)