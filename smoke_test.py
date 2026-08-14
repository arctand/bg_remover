from bgremover.config import load_config
from bgremover.inference import BiRefNetBackend
from bgremover.foreground import PyMattingForegroundRefiner
from PIL import Image

cfg = load_config()
print(BiRefNetBackend(cfg.model).smoke_test())
result = PyMattingForegroundRefiner(cfg.foreground).refine(
    Image.new("RGB", (32, 32), "white"), Image.new("L", (32, 32), 255)
)
print({"pymatting": result.rgb.size == result.alpha.size == (32, 32)})
