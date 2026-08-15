import numpy as np

from turboTensor import Tensor


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


def test_add_backward():

    x = Tensor(np.random.randn(3, 4))

    def f(x):
        return (x + x).sum()

    f(x).backward()

    numerical = numerical_gradient(f, x)

    assert np.allclose(
        x.grad,
        numerical,
        atol=1e-5,
        rtol=1e-5,
    )
    
def test_mul_backward():

    x = Tensor(np.random.randn(3, 4))
    weights = np.random.randn(3, 4)

    def f(x):
        return (x * Tensor(weights)).sum()

    f(x).backward()

    numerical = numerical_gradient(f, x)

    assert np.allclose(
        x.grad,
        numerical,
        atol=1e-5,
        rtol=1e-5,
    )

def test_matmul_backward():

    x = Tensor(np.random.randn(3, 4))
    w = Tensor(np.random.randn(4, 5))

    def f(x):
        return (x @ w).sum()

    f(x).backward()

    numerical = numerical_gradient(f, x)

    assert np.allclose(
        x.grad,
        numerical,
        atol=1e-5,
        rtol=1e-5,
    )
    
def test_logsumexp_backward():

    x = Tensor(np.random.randn(4, 5))

    def f(x):
        return x.logsumexp(axis=-1).mean()

    f(x).backward()

    numerical = numerical_gradient(f, x)

    assert np.allclose(
        x.grad,
        numerical,
        atol=1e-5,
        rtol=1e-5,
    )

def test_log_softmax_backward():

    x = Tensor(np.random.randn(4, 5))
    weights = Tensor(np.random.randn(4, 5))

    def f(x):
        return (x.log_softmax(axis=-1) * weights).sum()

    f(x).backward()

    numerical = numerical_gradient(f, x)

    assert np.allclose(
        x.grad,
        numerical,
        atol=1e-5,
        rtol=1e-5,
    )

def test_getitem_backward():

    x = Tensor(np.random.randn(5))

    indices = np.array([0, 2, 4])

    def f(x):
        return x[indices].sum()

    f(x).backward()

    numerical = numerical_gradient(f, x)

    assert np.allclose(
        x.grad,
        numerical,
        atol=1e-5,
        rtol=1e-5,
    )

def test_getitem_repeated_indices():

    x = Tensor(np.random.randn(3))

    y = x[[0, 0, 2]]

    y.sum().backward()

    assert np.allclose(
        x.grad,
        np.array([2.0, 0.0, 1.0]),
    )

