import numpy as np

from turboTensor import Tensor
from turboTensor.nn import (
    MSELoss,
    MAELoss,
    CrossEntropyLoss,
    BCELoss
)

def numerical_gradient(f, x, eps=1e-5):
    numerical = np.zeros_like(x.data)

    for idx in np.ndindex(x.data.shape):

        original = x.data[idx]

        x.data[idx] = original + eps
        plus = f(x).data

        x.data[idx] = original - eps
        minus = f(x).data

        x.data[idx] = original

        numerical[idx] = (plus - minus) / (2 * eps)

    return numerical

def test_cross_entropy_loss_value():
    """
    Verify CrossEntropyLoss against a manually computed value.
    """

    logits = Tensor(
        np.array([
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
        ])
    )

    target = np.array([0, 1])

    loss_fn = CrossEntropyLoss()

    loss = loss_fn(logits, target)

    # Manually compute:
    # CE = -mean(log(softmax(logits)[target]))
    exp_logits = np.exp(logits.data)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    expected = -np.mean(
        np.log([
            probs[0, 0],
            probs[1, 1],
        ])
    )

    assert np.allclose(
        loss.data,
        expected,
        atol=1e-7,
    )


def test_cross_entropy_backward():
    """
    Compare analytical gradient against numerical gradient.
    """

    np.random.seed(42)

    logits = Tensor(
        np.random.randn(4, 5)
    )

    target = np.array([0, 2, 1, 4])

    loss_fn = CrossEntropyLoss()

    # Analytical gradient
    loss = loss_fn(logits, target)
    loss.backward()

    analytical = logits.grad.copy()

    # Numerical gradient
    eps = 1e-5
    numerical = np.zeros_like(logits.data)

    for i in range(logits.data.shape[0]):
        for j in range(logits.data.shape[1]):

            original = logits.data[i, j]

            logits.data[i, j] = original + eps
            loss_plus = loss_fn(logits, target).data

            logits.data[i, j] = original - eps
            loss_minus = loss_fn(logits, target).data

            logits.data[i, j] = original

            numerical[i, j] = (
                loss_plus - loss_minus
            ) / (2 * eps)

    assert np.allclose(
        analytical,
        numerical,
        atol=1e-5,
        rtol=1e-5,
    )
    
def test_mse_loss_value():

    pred = Tensor(
        np.array([1.0, 2.0, 4.0])
    )

    target = Tensor(
        np.array([1.0, 3.0, 2.0])
    )

    loss_fn = MSELoss()

    loss = loss_fn(pred, target)

    expected = (
        (1 - 1) ** 2
        + (2 - 3) ** 2
        + (4 - 2) ** 2
    ) / 3

    assert np.allclose(
        loss.data,
        expected,
        atol=1e-7,
    )
    

def test_mse_loss_backward():

    pred = Tensor(
        np.random.randn(3, 4)
    )

    target = Tensor(
        np.random.randn(3, 4)
    )

    loss_fn = MSELoss()

    def f(x):
        return loss_fn(x, target)

    loss = f(pred)

    loss.backward()

    analytical = pred.grad.copy()

    numerical = numerical_gradient(
        f,
        pred,
    )

    assert np.allclose(
        analytical,
        numerical,
        atol=1e-5,
        rtol=1e-5,
    )
    

def test_mse_loss_value():

    pred = Tensor(
        np.array([1.0, 2.0, 4.0])
    )

    target = Tensor(
        np.array([1.0, 3.0, 2.0])
    )

    loss_fn = MSELoss()

    loss = loss_fn(pred, target)

    expected = (
        (1 - 1) ** 2
        + (2 - 3) ** 2
        + (4 - 2) ** 2
    ) / 3

    assert np.allclose(
        loss.data,
        expected,
        atol=1e-7,
    )
    
def test_mse_loss_backward():

    pred = Tensor(
        np.random.randn(3, 4)
    )

    target = Tensor(
        np.random.randn(3, 4)
    )

    loss_fn = MSELoss()

    def f(x):
        return loss_fn(x, target)

    loss = f(pred)

    loss.backward()

    analytical = pred.grad.copy()

    numerical = numerical_gradient(
        f,
        pred,
    )

    assert np.allclose(
        analytical,
        numerical,
        atol=1e-5,
        rtol=1e-5,
    )
    

def test_mae_loss_value():

    pred = Tensor(
        np.array([1.0, 2.0, 5.0])
    )

    target = Tensor(
        np.array([2.0, 4.0, 3.0])
    )

    loss_fn = MAELoss()

    loss = loss_fn(pred, target)

    expected = (
        abs(1 - 2)
        + abs(2 - 4)
        + abs(5 - 3)
    ) / 3

    assert np.allclose(
        loss.data,
        expected,
        atol=1e-7,
    )
    
def test_mae_loss_backward():

    pred = Tensor(
        np.random.randn(3, 4) + 1.0
    )

    target = Tensor(
        np.random.randn(3, 4) - 1.0
    )

    loss_fn = MAELoss()

    def f(x):
        return loss_fn(x, target)

    loss = f(pred)

    loss.backward()

    analytical = pred.grad.copy()

    numerical = numerical_gradient(
        f,
        pred,
    )

    assert np.allclose(
        analytical,
        numerical,
        atol=1e-5,
        rtol=1e-5,
    )

def test_bce_loss_value():

    pred = Tensor(
        np.array([0.8, 0.3, 0.9, 0.2])
    )

    target = Tensor(
        np.array([1.0, 0.0, 1.0, 0.0])
    )

    loss_fn = BCELoss()

    loss = loss_fn(pred, target)

    expected = -np.mean(
        target.data * np.log(pred.data)
        + (1 - target.data)
        * np.log(1 - pred.data)
    )

    assert np.allclose(
        loss.data,
        expected,
        atol=1e-7,
    )
    
def test_bce_loss_backward():

    pred = Tensor(
        np.array([
            [0.2, 0.7, 0.4],
            [0.8, 0.3, 0.6],
        ])
    )

    target = Tensor(
        np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ])
    )

    loss_fn = BCELoss()

    def f(x):
        return loss_fn(x, target)

    loss = f(pred)

    loss.backward()

    analytical = pred.grad.copy()

    numerical = numerical_gradient(
        f,
        pred,
    )

    assert np.allclose(
        analytical,
        numerical,
        atol=1e-5,
        rtol=1e-5,
    )
