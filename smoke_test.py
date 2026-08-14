from bgremover.config import load_config
from bgremover.inference import BiRefNetBackend

cfg = load_config()
print(BiRefNetBackend(cfg.model).smoke_test())
