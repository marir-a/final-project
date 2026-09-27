import tensorflow as tf
import numpy as np
import PIL.Image

from final.parameters.InitializationType import InitializationType


def initialize_image(init_type, content_image=None, style_image=None):
    """
    Initializes the target image with either content image, style image or noise depending on initialization type.
    :param init_type: a variable of InitializationType enum defining how to initialize the image.
    :param content_image: content image
    :param style_image: style image
    :return: a new initialized image
    """
    image = None
    match init_type:
        case InitializationType.CONTENT:
            image = tf.Variable(content_image)
        case InitializationType.STYLE:
            image = tf.Variable(style_image)
        case InitializationType.NOISE:
            image = tf.Variable(tf.random.uniform(shape=tf.shape(content_image), minval=0.0, maxval=1.0))
    return image


def tensor_to_image(tensor):
    """
    Converts a TensorFlow tensor into a PIL Image object.
    :param tensor: a tensor of shape [height, width, channels] or [batch_size, height, width, channels]
    :return: a PIL Image object.
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)
