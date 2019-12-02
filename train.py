import tensorflow as tf


def to_signed(binary_tensor):
    return 2 * binary_tensor - 1


def train(X, y, model, optimizer, num_iter=100):
    inputs = tf.convert_to_tensor(X, dtype=tf.float32)
    labels = to_signed(tf.convert_to_tensor(y, dtype=tf.float32))
    if labels.ndim == 1:
        labels = tf.expand_dims(labels, -1)

    loss_fn = lambda: model.loss(inputs, labels)
    var_list_fn = lambda: model.trainable_weights

    for _ in range(num_iter):
        optimizer.minimize(loss_fn, var_list_fn)

