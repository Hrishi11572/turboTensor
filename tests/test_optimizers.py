import numpy as np 
from turboTensor.optim import Adam, AdamW, SGD, Momentum
from turboTensor.nn import Parameter



def test_adam_first_step():

    p = Parameter(np.array([1.0]))

    optimizer = Adam(
        [p],
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
    )

    p.grad = np.array([0.5])

    optimizer.step()

    assert np.allclose(
        p.data,
        np.array([0.999]),
        atol=1e-7,
    )
    

def test_adamW_first_step():

    p = Parameter(np.array([1.0]))

    optimizer = AdamW(
        [p],
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.01
    )

    p.grad = np.array([0.5])

    optimizer.step()

    assert np.allclose(
        p.data,
        np.array([0.999]),
        atol=1e-7,
    )
    
def test_momentum_multiple_steps():

    p = Parameter(
        np.array([1.0])
    )

    optimizer = Momentum(
        [p],
        lr=0.1,
        momentum=0.9,
    )

    # Step 1
    p.grad = np.array([1.0])

    optimizer.step()

    # v1 = 0.9 * 0 + 1 = 1
    # p1 = 1 - 0.1 * 1 = 0.9

    assert np.allclose(
        p.data,
        np.array([0.9]),
        atol=1e-7,
    )

    assert np.allclose(
        optimizer.velocity[0],
        np.array([1.0]),
        atol=1e-7,
    )

    # Step 2
    p.grad = np.array([1.0])

    optimizer.step()

    # v2 = 0.9 * 1 + 1 = 1.9
    # p2 = 0.9 - 0.1 * 1.9 = 0.71

    assert np.allclose(
        p.data,
        np.array([0.71]),
        atol=1e-7,
    )

    assert np.allclose(
        optimizer.velocity[0],
        np.array([1.9]),
        atol=1e-7,
    )
    
def test_momentum_gradient_change():

    p = Parameter(
        np.array([1.0])
    )

    optimizer = Momentum(
        [p],
        lr=0.1,
        momentum=0.9,
    )

    # Step 1
    p.grad = np.array([1.0])
    optimizer.step()

    # v1 = 1
    # p1 = 0.9

    # Step 2: gradient changes
    p.grad = np.array([-1.0])
    optimizer.step()

    # v2 = 0.9(1) + (-1)
    #    = -0.1
    #
    # p2 = 0.9 - 0.1(-0.1)
    #    = 0.91

    assert np.allclose(
        optimizer.velocity[0],
        np.array([-0.1]),
        atol=1e-7,
    )

    assert np.allclose(
        p.data,
        np.array([0.91]),
        atol=1e-7,
    )
     
    
def test_sgd_single_step():
    """
    Verify the exact SGD update.
    """

    p = Parameter(
        np.array([1.0, 2.0, 3.0])
    )

    optimizer = SGD(
        [p],
        lr=0.1,
    )

    p.grad = np.array([
        0.5,
        -1.0,
        2.0,
    ])

    optimizer.step()

    expected = np.array([
        0.95,
        2.10,
        2.80,
    ])

    assert np.allclose(
        p.data,
        expected,
        atol=1e-7,
    )
    

def test_sgd_zero_grad():

    p = Parameter(
        np.array([1.0, 2.0])
    )

    optimizer = SGD(
        [p],
        lr=0.1,
    )

    p.grad = np.array([
        5.0,
        -3.0,
    ])

    optimizer.zero_grad()

    assert np.allclose(
        p.grad,
        np.zeros_like(p.grad),
    )