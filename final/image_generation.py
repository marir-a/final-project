import tensorflow as tf
from final.style_content_model import StyleContentModel
from final.helper_functions import initialize_image, tensor_to_image


def style_content_loss(outputs,
                       content_weight,
                       style_weight,
                       style_targets,
                       content_targets,
                       num_style_layers):
    """
    Calculates the weighted style loss and content loss, returns total loss.
    :param outputs: style and content activations of underlying classifier for target image
    :param content_weight: a coefficient for the content loss
    :param style_weight: a coefficient for the style loss
    :param style_targets: fixed values for the style layers activations of underlying classifier
    :param content_targets: fixed values for the content layers activations of underlying classifier
    :param num_style_layers: number of style layers used for style representation
    :return: a calculated total loss
    """
    style_outputs = outputs['style']
    content_outputs = outputs['content']

    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight

    loss = style_loss + content_loss
    return loss


@tf.function()
def train_step(image, run_parameters, extractor, style_targets, content_targets, opt):
    """
    Extracts the activations of generated images, calculates the loss and updates the pixels of the generated image.
    :param image: generated image
    :param run_parameters: parameters of the experiment
    :param extractor: StyleContentModel object that allows extraction of activations of underlying classifier
    :param style_targets: fixed values for the style layers activations of underlying classifier
    :param content_targets: fixed values for the content layers activations of underlying classifier
    :param opt: optimizer object
    """
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs,
                                  run_parameters.content_weight,
                                  run_parameters.style_weight,
                                  style_targets,
                                  content_targets,
                                  len(run_parameters.style_layer))
        loss += run_parameters.total_variation_weight * tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))


def run_generation(content_image, style_image, run_parameters, filename):
    """
    Run of an image generation. The image is initialized, target activations of content and style images are extracted
    and fixed, optimizer is initialized. For epochs and steps set in RunParameters object, the image is generated and
    then saved in a file.
    :param content_image: a content image
    :param style_image: a style image
    :param run_parameters: a set of parameters used for this image generation
    :param filename: a path to a file to save the generated image
    """
    image = initialize_image(run_parameters.initialization, content_image, style_image)
    content_layers = [run_parameters.content_layer.name]
    extractor = StyleContentModel(run_parameters.style_layer, content_layers)

    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    for n in range(run_parameters.epochs):
        for m in range(run_parameters.steps_per_epoch):
            train_step(image,
                       run_parameters,
                       extractor,
                       style_targets,
                       content_targets,
                       opt)
        im = tensor_to_image(image)
        im.save('results/{}/generation_{}.png'.format(filename, n))
