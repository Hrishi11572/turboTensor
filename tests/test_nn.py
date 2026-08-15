import numpy as np

from turboTensor import Tensor
from turboTensor.nn import (
    Linear,
    ReLU,
    Sequential,
    CrossEntropyLoss,
)
from turboTensor.nn import Parameter

    
# Tests Related to state_dict 

def test_state_dict():

    model = Sequential(
        Linear(10, 32),
        ReLU(),
        Linear(32, 5),
    )

    state = model.state_dict()

    assert set(state.keys()) == {
        "0.W",
        "0.b",
        "2.W",
        "2.b",
    }

    assert state["0.W"].shape == (10, 32)
    assert state["0.b"].shape == (32,)
    assert state["2.W"].shape == (32, 5)
    assert state["2.b"].shape == (5,)

def test_load_state_dict():

    model1 = Sequential(
        Linear(10, 32),
        ReLU(),
        Linear(32, 5),
    )

    model2 = Sequential(
        Linear(10, 32),
        ReLU(),
        Linear(32, 5),
    )

    state = model1.state_dict()

    model2.load_state_dict(state)

    params1 = model1.parameters()
    params2 = model2.parameters()

    assert len(params1) == len(params2)

    for p1, p2 in zip(params1, params2):
        assert np.allclose(
            p1.data,
            p2.data,
        )

def test_state_dict_does_not_share_parameter_memory():

    model = Sequential(
        Linear(10, 5),
    )

    state = model.state_dict()

    state["0.W"][0, 0] += 100

    assert not np.allclose(
        state["0.W"],
        model._modules["0"].W.data,
    )


def test_module_to_cpu():

    model = Sequential(
        Linear(10, 32),
        ReLU(),
        Linear(32, 5),
    )

    model.to("cpu")

    for parameter in model.parameters():
        assert parameter.device == "cpu"
        
def test_module_to_returns_self():

    model = Linear(10, 5)

    result = model.to("cpu")

    assert result is model
    

def test_nested_module_to_cpu():

    model = Sequential(
        Linear(10, 32),
        ReLU(),
        Sequential(
            Linear(32, 16),
            ReLU(),
            Linear(16, 5),
        ),
    )

    model.to("cpu")

    for parameter in model.parameters():
        assert parameter.device == "cpu"
        
        
def test_module_to_preserves_parameter_identity():

    model = Linear(10, 5)

    parameters_before = model.parameters()

    model.to("cpu")

    parameters_after = model.parameters()

    for before, after in zip(
        parameters_before,
        parameters_after,
    ):
        assert before is after