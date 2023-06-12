try:
    from nvtx import *
except ModuleNotFoundError:
    from .mock import *