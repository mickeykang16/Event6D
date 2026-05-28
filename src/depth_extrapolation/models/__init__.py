"""Model registry for Stage 1 (EventVFI) pretraining + inference.

Registers the model name → class mappings used in the configs. Each `from . import X`
side-effect-registers via the `@register(...)` decorator inside that module.
"""
from .models import register, make
from . import edsr      # noqa: F401  registers `edsr-baseline`
from . import mlp       # noqa: F401  registers `mlp`
from . import misc      # noqa: F401  registers utility models
from . import cbmnet_light_extrapolation_small_e2vid  # noqa: F401
