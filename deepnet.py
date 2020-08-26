import tensorflow as tf


def dense2layer(hidden):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(hidden, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


hinge_loss = tf.keras.losses.Hinge()
log_loss = lambda target, prediction: tf.keras.losses.BinaryCrossentropy()(
    target, tf.sigmoid(prediction)
)


class DeepNet2Layer(tf.keras.Model):
    def __init__(self, n_hidden, loss_func, regularization=0.0):
        super(DeepNet2Layer, self).__init__()
        self.regularizer = tf.keras.regularizers.l2(regularization)
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(
                    n_hidden, activation="relu", kernel_regularizer=self.regularizer
                ),
                tf.keras.layers.Dense(1, kernel_regularizer=self.regularizer),
            ]
        )
        self.loss_func = loss_func

    def call(self, input_tensor, training=False):
        return self.model(input_tensor)

    def pred(self, input_tensor):
        return tf.cast(tf.greater_equal(self(input_tensor), 0), tf.float32)

    def loss(self, input_tensor, label):
        output = self(input_tensor)
        return self.loss_func(label, output)

