import functools
import tensorflow as tf
import utils
from utils import assert_is_signs
import tensorflow_probability as tfp


def add_bias_feature(tensor):
    # tensor is of shape B x n
    return tf.pad(tensor, [[0, 0], [1, 0]], constant_values=1)


# def _make_val_and_grad_fn(value_fn):
#     @functools.wraps(value_fn)
#     def val_and_grad(x):
#         return tfp.math.value_and_gradient(value_fn, x)

#     return val_and_grad


# @_make_val_and_grad_fn
# def hinge(strength, label):
#     return tf.losses.hinge(label, strength)


class HebbNetNLayer(tf.keras.Model):
    def __init__(self, n_hiddens):
        super(HebbNetNLayer, self).__init__()
        print("This HebbNet has only 1 output!")
        self.regularizer = tf.keras.regularizers.l2(0.01)
        self.layer_list = [
            utils.SignLayer(hidden, kernel_regularizer=self.regularizer)
            for hidden in n_hiddens
        ] + [utils.SignLayer(1, kernel_regularizer=self.regularizer)]
        print("Not building layers!")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        # TODO: This is just to test out
        # self.call(tf.constant([[0.0, 0.0]]))
        # for layer in self.layer_list:
        # layer.kernel.assign(layer.kernel * 0)

    def call(self, input_tensor, training=False):
        output_dict = {"activations": [], "strengths": []}
        x = input_tensor
        for i, layer in enumerate(self.layer_list):
            # print(f"{i} before bias add: {x.shape}")
            # x = add_bias_feature(x)
            # print(f"{i} after bias add: {x.shape}")
            x, s = layer(x)
            print(f"{i} layer out: {x.shape}")
            output_dict["activations"].append(x)
            output_dict["strengths"].append(s)

        return output_dict

    def pred(self, input_tensor):
        return self(input_tensor)["activations"][-1]

    def update(self, input_tensor, label):
        # label is B x 1
        # input_tensor is B x n
        assert_is_signs(label)

        output_dict = self(input_tensor)
        output = output_dict["activations"][-1]
        print(f"Loss: {self.loss(output, label)}")

        for layer, strength in zip(
            reversed(self.layers), reversed(output_dict["strengths"])
        ):
            loss_fn = lambda: tf.losses.hinge(label, strength) + layer.losses
            var_list_fn = lambda: [layer.kernel, layer.bias]

            for _ in range(100):
                self.optimizer.minimize(loss_fn, var_list_fn)

            # weight_bias = tf.nn.l2_normalize(
            #     weight_bias, axis=1
            # )  # weight is out x in + 1
            # weight, bias = weight_bias[:, 1:], weight_bias[:, 0]
            # layer.kernel = tf.transpose(weight)  # kernel is in x out
            # layer.bias = bias

            # new label: should be weighted by weight?
            label = tf.matmul(label, tf.stop_gradient(tf.transpose(layer.kernel)))

    @staticmethod
    def loss(activation, label):
        # label is B x n
        # activation is  B x n
        return tf.reduce_mean(-label * activation)

