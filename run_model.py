import numpy as np
import tensorflow as tf

import hebbnet
import decision_boundary
from train import train


X, y = decision_boundary.generate_blobs(show=False)

n_samples = X.shape[0]
n_features = X.shape[1]
n_classes = len(np.unique(y))

assert n_classes == 2

loss_fn = hebbnet.create_loss_fn(hebbnet.loss_a, hebbnet.base_loss_max_margin)
# model = hebbnet.HebbNetSimple(loss_fn, regularization=0.0)
model = hebbnet.HebbNet2Layer(500, loss_fn, regularization=0.01)

# Higher learning rates will converge faster, but it might not converge as well or it might diverge or bounce around
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

train(X, y, model, optimizer, num_iter=300)


def to_numpy_pred_func(pred_func):
    def new_pred_func(np_inputs):
        tensor_inputs = tf.convert_to_tensor(np_inputs)
        return pred_func(tensor_inputs).numpy()

    return new_pred_func


decision_boundary.plot_decision_boundary(to_numpy_pred_func(model.pred), X, y)

