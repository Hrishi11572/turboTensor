from .numpy_backend import NumPyBackend

default_backend = NumPyBackend()

def get_backend(device):

    if device == "cpu":
        return default_backend

    if device == "cuda":
        from .cupy_backend import CuPyBackend
        return CuPyBackend()

    raise ValueError(
        f"Unknown device: {device}"
    )
    
    
__all__ = [
    "NumPyBackend",
    "default_backend",
]