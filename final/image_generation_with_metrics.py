import pandas as pd
import tensorflow as tf

from final.style_content_model import StyleContentModel
from final.helper_functions import tensor_to_image, initialize_image


def style_content_loss_with_metrics(outputs,
                                    content_weight,
                                    style_weight,
                                    style_targets,
                                    content_targets,
                                    num_style_layers):
    """
    Calculates the weighted style loss and content loss, returns raw and weighted content losses and style losses.
    :param outputs: style and content activations of underlying classifier for target image
    :param content_weight: a coefficient for the content loss
    :param style_weight: a coefficient for the style loss
    :param style_targets: fixed values for the style layers activations of underlying classifier
    :param content_targets: fixed values for the content layers activations of underlying classifier
    :param num_style_layers: number of style layers used for style representation
    :return: weighted content loss, weighted style loss, raw content loss, raw style loss
    """
    style_outputs = outputs['style']
    content_outputs = outputs['content']

    raw_style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                               for name in style_outputs.keys()]) / num_style_layers

    raw_content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                                 for name in content_outputs.keys()])

    style_loss = style_weight * raw_style_loss
    content_loss = content_weight * raw_content_loss

    return content_loss, style_loss, raw_content_loss, raw_style_loss


@tf.function()
def train_step_with_metrics(image, run_parameters, extractor, style_targets, content_targets, opt):
    """
    Extracts the activations of generated images, calculates the loss and updates the pixels of the generated image.
    Keeps the information about losses and gradients with respect to content loss, style loss, total variation loss.
    :param image: generated image
    :param run_parameters: parameters of the experiment
    :param extractor: StyleContentModel object that allows extraction of activations of underlying classifier
    :param style_targets: fixed values for the style layers activations of underlying classifier
    :param content_targets: fixed values for the content layers activations of underlying classifier
    :param opt: optimizer object
    :return: a dictionary with different metrics such as losses, gradients, etc.
    """
    with tf.GradientTape(persistent=True) as tape:  # used persistent=True to make a several .gradient() calls
        outputs = extractor(image)

        content_loss, style_loss, raw_content_loss, raw_style_loss = style_content_loss_with_metrics(
            outputs,
            run_parameters.content_weight,
            run_parameters.style_weight,
            style_targets,
            content_targets,
            len(run_parameters.style_layer)
        )

        total_variation_loss = run_parameters.total_variation_weight * tf.image.total_variation(image)
        total_loss = content_loss + style_loss + total_variation_loss

    content_grad = tape.gradient(content_loss, image)
    style_grad = tape.gradient(style_loss, image)
    total_variation_grad = tape.gradient(total_variation_loss, image)

    total_grad = tape.gradient(total_loss, image)

    content_grad_norm = tf.norm(content_grad)
    style_grad_norm = tf.norm(style_grad)
    total_variation_grad_norm = tf.norm(total_variation_grad)

    opt.apply_gradients([(total_grad, image)])

    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

    del tape

    return {
        "total_loss": total_loss,
        "content_loss": content_loss,
        "style_loss": style_loss,
        "raw_content_loss": raw_content_loss,
        "raw_style_loss": raw_style_loss,
        "content_grad_norm": content_grad_norm,
        "style_grad_norm": style_grad_norm,
        "total_variation_grad_norm": total_variation_grad_norm
    }


def save_metrics(history, metrics, iteration):
    """
    Saves metrics to a history dictionary.
    :param history: a dictionary with a history of metrics values.
    :param metrics: a dictionary with new values to be added to history.
    :param iteration: an iteration number.
    """
    history["step"].append(iteration)
    history["total_loss"].append(metrics["total_loss"])
    history["content_loss"].append(float(metrics["content_loss"]))
    history["style_loss"].append(float(metrics["style_loss"]))
    history["raw_content_loss"].append(float(metrics["raw_content_loss"]))
    history["raw_style_loss"].append(float(metrics["raw_style_loss"]))
    history["content_grad_norm"].append(float(metrics["content_grad_norm"]))
    history["style_grad_norm"].append(float(metrics["style_grad_norm"]))


def run_generation_with_metrics(content_image, style_image, run_parameters, filename, metrics_filename):
    """
        Run of an image generation. The image is initialized, target activations of content and style images are extracted
        and fixed, optimizer is initialized. For epochs and steps set in RunParameters object, the image is generated and
        then saved in a file. Metrics such as content_loss, style_loss and others saved in a CSV file.
        :param content_image: a content image
        :param style_image: a style image
        :param run_parameters: a set of parameters used for this image generation
        :param filename: a path to a file to save the generated image
        :param metrics_filename: a path to a file to save the generated metrics
        """
    image = initialize_image(run_parameters.initialization, content_image, style_image)
    content_layers = [run_parameters.content_layer.name]
    extractor = StyleContentModel(run_parameters.style_layer, content_layers)

    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    history = {
        "step": [],
        "total_loss": [],
        "content_loss": [],
        "style_loss": [],
        "raw_content_loss": [],
        "raw_style_loss": [],
        "content_grad_norm": [],
        "style_grad_norm": []
    }

    for n in range(run_parameters.epochs):
        for m in range(run_parameters.steps_per_epoch):
            metrics = train_step_with_metrics(image,
                                              run_parameters,
                                              extractor,
                                              style_targets,
                                              content_targets,
                                              opt)
            if m % 10 == 0:
                save_metrics(history, metrics, m)
        im = tensor_to_image(image)
        im.save('results/{}/generation_{}.png'.format(filename, n))
    df = pd.DataFrame(history)
    df.to_csv("results/{}/metrics.csv".format(metrics_filename), index=False)
