from enum import Enum


class InitializationType(Enum):
    """
    Enum that lists initialization of generated image for different experiment setups.
    Parameter sets 1, 4, 5 - CONTENT;
    Parameter set 2 - STYLE;
    Parameter set 3 - NOISE.
    """
    CONTENT = 1,
    STYLE = 2,
    NOISE = 3,
