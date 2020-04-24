import numpy as np
from reskit.core.util.loss_factors import low_generation_loss


def test_low_generation_loss():
    assert np.isclose(low_generation_loss(0.05, base=0, sharpness=5), 0.7788007830714049)
    assert np.isclose(low_generation_loss(0.05, base=0.5, sharpness=5), 0.38940039153570244)
    assert np.isclose(low_generation_loss(0.05, base=0.3, sharpness=5), 0.5451605481499834)
    assert np.isclose(low_generation_loss(0.25, base=0.3, sharpness=20), 0.004716562899359827)
    assert np.isclose(low_generation_loss(0.50, base=0.5, sharpness=1), 0.3032653298563167)
