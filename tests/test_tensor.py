import numpy as np

from turboTensor.nn import (
    Parameter,
    Linear,
    Sequential,
    ReLU,
)

from turboTensor.optim import Adam
from turboTensor.tensor import Tensor
from turboTensor.nn import CrossEntropyLoss

def test_linear_parameters():

    layer = Linear(10, 5)

    params = layer.parameters()

    assert len(params) == 2

    assert all(
        isinstance(p, Parameter)
        for p in params
    )

    assert params[0].data.shape == (10, 5)
    assert params[1].data.shape == (5,)


def test_sequential_parameters():

    model = Sequential(
        Linear(10, 32),
        ReLU(),
        Linear(32, 5),
    )

    params = model.parameters()

    assert len(params) == 4

    assert params[0].data.shape == (10, 32)
    assert params[1].data.shape == (32,)
    assert params[2].data.shape == (32, 5)
    assert params[3].data.shape == (5,)
    

def test_training_step():

    np.random.seed(42)

    model = Sequential(
        Linear(10, 32),
        ReLU(),
        Linear(32, 5),
    )

    optimizer = Adam(
        model.parameters(),
        lr=1e-3,
    )

    X = Tensor(np.random.randn(16, 10))
    y = np.random.randint(0, 5, size=16)

    loss_fn = CrossEntropyLoss()

    logits = model(X)
    loss_before = loss_fn(logits, y)

    optimizer.zero_grad()
    loss_before.backward()
    optimizer.step()

    logits = model(X)
    loss_after = loss_fn(logits, y)

    assert np.isfinite(loss_before.data)
    assert np.isfinite(loss_after.data)