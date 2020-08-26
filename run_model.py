import numpy as np
import tensorflow as tf
from sklearn.svm import LinearSVC
import sklearn.linear_model

import hebbnet
import deepnet
import hebbnet2
import decision_boundary
from train import train, train2


X, y = decision_boundary.generate_xor(signed=True)

n_samples = X.shape[0]
n_features = X.shape[1]
n_classes = len(np.unique(y))

assert n_classes == 2, f"Found {n_classes} classes"

# loss_fn = hebbnet.create_loss_fn(hebbnet.loss_c, hebbnet.base_loss_max_margin)
# model = hebbnet.HebbNetSimple(loss_fn, regularization=0.0)
# model = hebbnet.HebbNetNLayer([200], loss_fn, regularization=0.0)
# model = hebbnet.HebbNetNLayer([100, 100, 100], loss_fn, regularization=0.0)
# model = hebbnet.HebbNetNLayer([200], loss_fn, regularization=0.0)
# model = deepnet.DeepNet2Layer(200, deepnet.hinge_loss)

# model = hebbnet2.HebbNetNLayer([2])

# model = LinearSVC(C=1000.0, max_iter=10000)
model = sklearn.linear_model.LogisticRegression()

# Higher learning rates will converge faster, but it might not converge as well or it might diverge or bounce around
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

# train(X, y, model, optimizer, num_iter=1000)
# train2(X, y, model, num_iter=10)
model.fit(X, y)
# model.coef_ = np.array([[1, 1]])
# model.intercept_ = np.array([-1.5])


def to_numpy_pred_func(pred_func):
    def new_pred_func(np_inputs):
        tensor_inputs = tf.convert_to_tensor(np_inputs)
        return pred_func(tensor_inputs).numpy()

    return new_pred_func


# decision_boundary.plot_decision_boundary(to_numpy_pred_func(model.pred), X, y)
decision_boundary.plot_decision_boundary(model.predict, X, y)
print(f"Coeffecients: {model.coef_}  : bias : {model.intercept_}")

