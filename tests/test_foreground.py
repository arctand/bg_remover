import numpy as np
from PIL import Image

from bgremover.config import ForegroundConfig
from bgremover.foreground import PyMattingForegroundRefiner


def test_pymatting_preserves_dimensions_and_alpha():
    height, width = 24, 32
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = 180
    alpha = np.zeros((height, width), dtype=np.uint8)
    alpha[4:20, 6:26] = 255
    alpha[3, 6:26] = 96
    matte = Image.fromarray(alpha)

    result = PyMattingForegroundRefiner(ForegroundConfig()).refine(Image.fromarray(rgb), matte)

    assert result.rgb.size == result.alpha.size == (width, height)
    assert np.array_equal(np.asarray(result.alpha), alpha)
