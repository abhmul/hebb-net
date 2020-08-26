import tensorflow as tf


def assert_is_signs(tensor):
    tf.Assert(tf.reduce_all((tensor == 1) | (tensor == -1)), [tensor])


def assert_is_binary(tensor):
    tf.Assert(tf.reduce_all((tensor == 0) | (tensor == 1)), [tensor])


def heaveside(x: tf.Tensor):
    return tf.stop_gradient(tf.cast(tf.greater_equal(x, 0), tf.float32))


class SignLayer(tf.keras.layers.Dense):
    def call(self, x):
        strength = super().call(x)
        # sign will return 0 if strength is 0
        activation = tf.stop_gradient(tf.sign(strength))
        return activation, strength


class HeavesideLayer(tf.keras.layers.Dense):
    def call(self, x):
        strength = super().call(x)
        activation = heaveside(strength)
        return activation, strength