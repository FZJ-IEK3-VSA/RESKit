import numpy as np


def low_generation_loss(capacity_factor, base=0, sharpness=5):
    """Generate capacity-factor-dependent loss factors

    Follows the equation:
        (1-base) * ( 1 - exp[-sharpness * capacity_factor] )
    """
    return (1 - base) * (1 - np.exp(-sharpness * capacity_factor))  # dampens lower wind speeds
