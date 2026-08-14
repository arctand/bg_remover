from bgremover.config import load_config
from bgremover.diagnostics import diagnostics_text

print(diagnostics_text(load_config()))
