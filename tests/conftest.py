import sys
import os

try:
    import msvc_runtime
except ImportError:
    pass

try:
    import torch
except Exception:
    pass
