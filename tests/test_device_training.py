import numpy as np
import pytest 
from turboTensor import Tensor
from turboTensor.nn import (
    Linear,
    ReLU,
    Sequential,
    CrossEntropyLoss,
)
from turboTensor.optim import Adam


def test_cpu_training():

    np.random.seed(42)

    model = Sequential(
        Linear(10, 32),
        ReLU(),
        Linear(32, 5),
    )

    model.to("cpu")

    optimizer = Adam(
        model.parameters(),
        lr=1e-3,
    )

    X = Tensor(
        np.random.randn(16, 10)
    ).to("cpu")

    y = np.random.randint(
        0,
        5,
        size=16,
    )

    loss_fn = CrossEntropyLoss()

    logits = model(X)
    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert np.isfinite(loss.data)

    for parameter in model.parameters():
        assert parameter.device == "cpu"
        
def test_cuda_training():

    cp = pytest.importorskip("cupy")

    model = Sequential(
        Linear(10, 32),
        ReLU(),
        Linear(32, 5),
    )

    model.to("cuda")

    X = Tensor(
        np.random.randn(16, 10)
    ).to("cuda")

    y = cp.asarray(
        np.random.randint(
            0,
            5,
            size=16,
        )
    )

    loss_fn = CrossEntropyLoss()

    logits = model(X)

    assert isinstance(
        logits.data,
        cp.ndarray,
    )

    loss = loss_fn(logits, y)

    loss.backward()

    for parameter in model.parameters():

        assert isinstance(
            parameter.data,
            cp.ndarray,
        )

        assert isinstance(
            parameter.grad,
            cp.ndarray,
        )