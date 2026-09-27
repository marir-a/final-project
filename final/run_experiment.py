from tensorflow.python.distribute.coordinator.metric_utils import enable_metrics

from final.image_generation import run_generation
from final.image_generation_with_metrics import run_generation_with_metrics
from final.parameters.ContentLayer import ContentLayer
from final.parameters.InitializationType import InitializationType
from final.parameters.RunParameters import RunParameters
from final.read_images import read_files_batch_for_experiment_run


def run_experiment(run_parameters, content_images_folder, style_images_folder, enable_metrics=False):
    """
    Run an experiment with the given parameters for the given content images and style images.
    :param run_parameters: a RunParameters object containing the parameter set for the experiment run
    :param content_images_folder: a folder containing content images
    :param style_images_folder: a folder containing style images
    :param enable_metrics: a flag to enable or disable metrics collection
    """
    image_pairs = read_files_batch_for_experiment_run(content_images_folder, style_images_folder)
    i = 0
    for image_pair in image_pairs:
        if enable_metrics:
            run_generation_with_metrics(image_pair[0],
                                        image_pair[1],
                                        run_parameters,
                                        'cont' + str(i) + 'style',
                                        "experiment1")
        else:
            run_generation(image_pair[0],
                           image_pair[1],
                           run_parameters,
                           'cont' + str(i) + 'style')
        i += 1


### Put code here to generate and run the experiment. Example:
run_parameters = RunParameters(1e-2, 10, InitializationType.NOISE, ContentLayer.block4_conv2)
run_experiment(run_parameters, 'img/content/portraits', 'img/styles', True)
