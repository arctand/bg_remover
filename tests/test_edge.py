import numpy as np
from PIL import Image

from bgremover.config import EdgeConfig
from bgremover.edge import decontaminate_rgb


def test_decontamination_reduces_background_spill_without_changing_alpha():
    alpha=np.zeros((40,40),np.uint8); alpha[8:32,8:32]=255; alpha[7,8:32]=128; alpha[32,8:32]=128; alpha[8:32,7]=128; alpha[8:32,32]=128
    # Red foreground composited over green background creates green contaminated edge.
    a=alpha[...,None]/255.0; fg=np.full((40,40,3),(220,30,20),np.float32); bg=np.full((40,40,3),(10,220,30),np.float32)
    composite=np.rint(fg*a+bg*(1-a)).astype(np.uint8)
    matte=Image.fromarray(alpha); corrected,metrics=decontaminate_rgb(Image.fromarray(composite),matte,EdgeConfig())
    out=np.asarray(corrected); edge=alpha==128
    assert out[edge,1].mean() < composite[edge,1].mean()
    assert metrics.edge_pixels > 0 and np.array_equal(np.asarray(matte),alpha)
