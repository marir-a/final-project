from enum import Enum


class ContentLayer(Enum):
    """
    Enum that lists content layers for different experiment setups.
    Parameter sets 1, 2, 3, 4 - block5_conv2;
    Parameter set 5 - block4_conv2.
    """
    block5_conv2 = 'block5_conv2',
    block4_conv2 = 'block4_conv2',
