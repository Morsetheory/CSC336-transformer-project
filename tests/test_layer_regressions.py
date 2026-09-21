import pytest
import torch

from cs336_basics.layers import Embedding, Linear, SwiGLU, rmsnorm


@pytest.mark.parametrize("shape", [(2048, 64), (256, 512)])
def test_embedding_initialization_has_unit_scale_and_three_sigma_bounds(shape):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        weight = Embedding(*shape).weight.detach()

    # A standard normal truncated at three sigma has standard deviation ~0.987.
    assert abs(weight.mean().item()) < 0.015
    assert 0.97 < weight.std().item() < 1.01
    assert -3.0 <= weight.min().item() < -2.9
    assert 2.9 < weight.max().item() <= 3.0


LAYER_FACTORIES = [
    pytest.param(lambda **kwargs: Linear(4, 3, **kwargs), id="linear"),
    pytest.param(lambda **kwargs: Embedding(8, 4, **kwargs), id="embedding"),
    pytest.param(lambda **kwargs: rmsnorm(4, **kwargs), id="rmsnorm"),
    pytest.param(lambda **kwargs: SwiGLU(4, 8, **kwargs), id="swiglu"),
]


@pytest.mark.parametrize("factory", LAYER_FACTORIES)
def test_layer_constructor_float64_supports_forward_and_backward(factory):
    layer = factory(device="cpu", dtype=torch.float64)
    assert all(parameter.dtype == torch.float64 for parameter in layer.parameters())

    if isinstance(layer, Embedding):
        inputs = torch.tensor([[0, 2], [1, 7]])
    else:
        inputs = torch.randn(2, 4, dtype=torch.float64)
    output = layer(inputs)
    assert output.dtype == torch.float64
    assert torch.isfinite(output).all()

    output.square().sum().backward()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert parameter.grad.dtype == torch.float64
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("factory", LAYER_FACTORIES)
def test_layer_constructor_places_all_parameters_on_requested_device(factory):
    layer = factory(device="meta", dtype=torch.float64)
    for parameter in layer.parameters():
        assert parameter.device.type == "meta"
        assert parameter.dtype == torch.float64
