import os
import tensorflow as tf


def read_files_batch_for_experiment_run(content_images_folder, style_images_folder):
    """
    Reads content and style images from content_images_folder, converts them to tensors and saves pairs in a list.
    :param content_images_folder: a folder containing content images
    :param style_images_folder: a folder containing style images
    :return: a list of pairs of tensors: content_tensor, style_tensor
    """
    pairs = []
    for image in os.listdir(content_images_folder):
        content_path = content_images_folder + "/" + image
        content_img, shape = load_img(content_path)
        for style in os.listdir(style_images_folder):
            style_path = style_images_folder + "/" + style
            style_img, _ = load_img(style_path, content_height=shape[0], content_width=shape[1])
            print(style_img.shape)
            pairs.append((content_img, style_img))
    return pairs


def load_img(path_to_img, max_dimension=512, content_height=None, content_width=None):
    """
    Loads image from path_to_img, resizes it to content height / content width if they are provided, scales down
    to max dimension, converts to TensorFlow tensor and returns it.
    :param path_to_img: a filepath to the image
    :param max_dimension: maximum dimension of the image
    :param content_height: height of content image (when loading style image, its height and width are resized to content image dimensions)
    :param content_width: width of content image (when loading style image, its height and width are resized to content image dimensions)
    :return:
    """
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    if content_height is not None and content_width is not None:
        img = tf.image.resize(img, [content_height, content_width])

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dimension / long_dim

    new_shape = tf.cast(tf.round(shape * scale), tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    print(new_shape)
    return img, new_shape
