from final.parameters.ContentLayer import ContentLayer
from final.parameters.InitializationType import InitializationType


class RunParameters:
    """
    Parameters for one set of image generations.
    Variable parameters are initialized with __init__ function, constant parameters are initialized with values.
    """

    def __init__(self, style_weight, content_weight, initialization: InitializationType, content_layer: ContentLayer):
        """
        Parameter set initialization. Constant values:
        style_layers: style layers chosen for style representation
        epochs: number of epochs for image generation
        steps_per_epoch: number of steps per epoch for image generation
        total_variation_weight: value used for the loss function computation that defines the weight of total variation loss

        :param style_weight: alpha value used for the loss function computation that defines the weight of style loss
        :param content_weight: beta value used for the loss function computation that defines the weight of content loss
        :param initialization: generated image initialization: content, style or noise
        :param content_layer: content layer chosen for content representation
        """
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.initialization = initialization
        self.content_layer = content_layer
        self.style_layer = ['block1_conv1',
                            'block2_conv1',
                            'block3_conv1',
                            'block4_conv1',
                            'block5_conv1']
        self.epochs = 5
        self.steps_per_epoch = 100
        self.total_variation_weight = 20
