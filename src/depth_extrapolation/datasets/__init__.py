"""Datasets registry for Stage 1 (EventVFI) pretraining."""
from .datasets import register, make
from . import image_folder  # noqa: F401  registers blender-train-wo-mask-recon-single
from . import wrappers      # noqa: F401  registers blender-e2vid-* wrappers
