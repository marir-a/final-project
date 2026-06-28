from keras.utils import load_img, img_to_array
import numpy as np
from keras.applications import vgg19
import tensorflow as tf

content_reference_image_path = 'img/revan.jpg'  # path to content image
style_reference_image_path = 'img/Starry_Night.jpg'  # path to style image
width, height = load_img(content_reference_image_path).size  # dimensions of the generated picture
img_height = 450
img_width = int(width * img_height / height)


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))  # loading image
    img = img_to_array(img)  # make it (H x W x C array) e.g. (400, 400, 3)
    img = np.expand_dims(img, axis=0)  # make it (1 x H x W x C) e.g. (1, 400, 400, 3)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


content_image_tf = tf.constant(preprocess_image(content_reference_image_path))
style_image_tf = tf.constant(preprocess_image(style_reference_image_path))

combination_image_tf = tf.Variable(
    initial_value=content_image_tf,
    trainable=True,
    dtype=tf.float32,
)

vgg = tf.keras.applications.VGG19(
    include_top=False,
    weights="imagenet",
)
vgg.trainable = False


# features_vgg = vgg(input_tensor, training=False)

def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))


#
def gram_matrix(x):
    # Expects x with shape: (height, width, channels)
    features = tf.transpose(x, perm=(2, 0, 1))  # (C, H, W)
    features = tf.reshape(features, [tf.shape(features)[0], -1])  # (C, H*W)
    return tf.matmul(features, features, transpose_b=True)  # (C, C)


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return tf.reduce_sum(tf.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(x):
    a = tf.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, 1:, :img_width - 1, :])
    b = tf.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, :img_height - 1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))


content_layer = "block5_conv2"
style_layers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

total_variation_weight = 1e-4
style_weight = 1
content_weight = 0.2

# Build the feature-extraction model once.
outputs_dict = {
    layer.name: layer.output
    for layer in vgg.layers
}

feature_extractor = tf.keras.Model(
    inputs=vgg.input,
    outputs=outputs_dict,
)


def compute_loss():
    # combination_image_tf changes each time SciPy calls Evaluator.loss().
    input_tensor = tf.concat(
        [content_image_tf, style_image_tf, combination_image_tf],
        axis=0,
    )

    # All images are already preprocessed by preprocess_image().
    all_features = feature_extractor(input_tensor, training=False)

    loss = tf.constant(0.0, dtype=tf.float32)

    # Content loss
    layer_features = all_features[content_layer]

    content_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]

    loss += content_weight * content_loss(
        content_image_features,
        combination_features,
    )

    # Style loss
    for layer_name in style_layers:
        layer_features = all_features[layer_name]

        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]

        sl = style_loss(
            style_reference_features,
            combination_features,
        )

        loss += (style_weight / len(style_layers)) * sl

    # Total variation loss
    loss += total_variation_weight * total_variation_loss(
        combination_image_tf
    )

    return loss


def get_loss_and_grads(x):
    # SciPy supplies a flattened float64 NumPy vector.
    x = np.asarray(x, dtype=np.float32)
    x = x.reshape((1, img_height, img_width, 3))

    # Put SciPy's proposed image into the TensorFlow variable.
    combination_image_tf.assign(x)

    with tf.GradientTape() as tape:
        loss_value = compute_loss()

    grad_values = tape.gradient(loss_value, combination_image_tf)

    if grad_values is None:
        raise RuntimeError("Could not compute gradients.")

    return (
        float(loss_value.numpy()),
        grad_values.numpy().flatten().astype("float64"),
    )


class Evaluator:
    def __init__(self):
        self.loss_value = None
        self.grad_values = None

    def loss(self, x):
        assert self.loss_value is None

        self.loss_value, self.grad_values = get_loss_and_grads(x)

        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None

        grad_values = np.copy(self.grad_values)

        self.loss_value = None
        self.grad_values = None

        return grad_values


evaluator = Evaluator()

from scipy.optimize import fmin_l_bfgs_b
from keras.utils import save_img
import time

result_prefix = "my_result"
iterations = 20

x = combination_image_tf.numpy().flatten().astype("float64")

for i in range(iterations):
    print("Start of iteration", i)
    start_time = time.time()

    x, min_val, info = fmin_l_bfgs_b(
        func=evaluator.loss,
        x0=x,
        fprime=evaluator.grads,
        maxfun=20,
    )

    print("Current loss value:", min_val)

    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img.copy())

    fname = f"{result_prefix}_at_iteration_{i}.png"
    save_img(fname, img)

    end_time = time.time()
    print(f"Image saved as {fname}")
    print(f"Iteration {i} completed in {int(end_time - start_time)}s")
