from turboTensor.backend.numpy_backend import NumPyBackend
from turboTensor import Tensor
import numpy as np 

def test_numpy_backend():

    backend = NumPyBackend()

    x = backend.asarray([1, 2, 3])

    assert np.allclose(
        backend.exp(x),
        np.exp(x),
    )
    
def test_tensor_uses_backend():

    backend = NumPyBackend()

    x = Tensor(
        np.array([1.0, 2.0, 3.0]),
        backend=backend,
    )

    assert x.backend is backend

def test_backward_preserves_backend():

    backend = NumPyBackend()

    x = Tensor(
        np.array([1.0, 2.0, 3.0]),
        backend=backend,
    )

    y = (x * x).sum()

    y.backward()

    assert x.backend is backend
    
    
def test_tensor_device():

    x = Tensor(
        np.array([1.0, 2.0, 3.0])
    )

    assert x.device == "cpu"
    

def test_tensor_to_same_device():

    x = Tensor(
        np.array([1.0, 2.0, 3.0])
    )

    y = x.to("cpu")

    assert y.device == "cpu"
    assert np.allclose(
        y.data,
        x.data,
    )

import pytest

def test_tensor_to_cuda():

    cp = pytest.importorskip("cupy")

    x = Tensor(
        np.array([1.0, 2.0, 3.0])
    )

    y = x.to("cuda")

    assert y.device == "cuda"
    assert isinstance(y.data, cp.ndarray)
    
def test_tensor_to_cpu():

    x = Tensor(
        np.array([1.0, 2.0, 3.0])
    )

    y = x.to("cpu")

    assert y.device == "cpu"
    assert np.allclose(y.data, x.data)
    
def test_tensor_to_does_not_modify_original():

    x = Tensor(
        np.array([1.0, 2.0, 3.0])
    )

    y = x.to("cpu")

    assert x.device == "cpu"
    assert y.device == "cpu"