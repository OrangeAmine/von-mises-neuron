"""Independent numerical references for the new float64 locality package."""

import copy
import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.locality_models import (
    CircularBasisV2,
    chordal_features,
    circular_features,
    fourier_features,
    make_basis,
    predict_readout,
    readout_diagnostics,
    reduced_objective,
    solve_readout,
)


def geometry():
    """Return explicit geometry without consuming any global RNG."""
    return (torch.tensor([-2.2, -0.5, 0.3, 1.7], dtype=torch.float64),
            torch.tensor([0.4, 2.0, 4.0, 19.0], dtype=torch.float64))


def example():
    """Return a nontrivial deterministic scalar regression example."""
    theta = torch.linspace(-3.05, 3.1, 37, dtype=torch.float64)
    target = 2 + theta.sin() - 0.3 * (3 * theta).cos() + 0.08 * (7 * theta).sin()
    return theta, target


@pytest.mark.parametrize("mode,learn_centers,learn_widths", [
    ("fixed", False, False), ("centers", True, False),
    ("widths", False, True), ("both", True, True),
])
def test_basis_explicit_flags_no_aliases_rng_or_input_autograd(mode, learn_centers, learn_widths):
    centers, widths = [value.requires_grad_() for value in geometry()]
    before = torch.get_rng_state().clone()
    basis = make_basis(centers, widths, mode)
    assert torch.equal(before, torch.get_rng_state())
    assert basis.version == "CircularBasisV2"
    assert basis.centers.shape == widths.shape == basis.concentrations.shape
    assert basis.centers.dtype == basis.concentrations.dtype == torch.float64
    assert basis.centers.device.type == "cpu"
    assert basis.centers.requires_grad == learn_centers
    assert basis.concentrations.requires_grad == learn_widths
    assert sum(p.numel() for p in basis.parameters()) == 4 * (learn_centers + learn_widths)
    assert ("bank.centers" in dict(basis.named_parameters())) == learn_centers
    assert ("bank.raw_kappa" in dict(basis.named_parameters())) == learn_widths
    assert ("bank.fixed_kappa" in dict(basis.named_buffers())) != learn_widths
    assert basis.centers.data_ptr() != centers.data_ptr()
    assert basis.concentrations.data_ptr() != widths.data_ptr()
    torch.testing.assert_close(basis.concentrations, widths.detach(), atol=1e-14, rtol=1e-14)
    assert basis.get_config()["mode"] == mode
    theta, _ = example()
    original = basis(theta).detach().clone()
    with torch.no_grad():
        centers.add_(2)
        widths.mul_(3)
    torch.testing.assert_close(basis(theta), original, atol=0, rtol=0)
    if learn_centers or learn_widths:
        basis(theta).sum().backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in basis.parameters())
    assert centers.grad is None and widths.grad is None
    clone = make_basis(*geometry(), mode)
    clone.load_state_dict(copy.deepcopy(basis.state_dict()))
    torch.testing.assert_close(clone(theta), original, atol=0, rtol=0)


def test_list_and_column_initialization_converts_to_float64_without_default_dtype_change():
    old_dtype = torch.get_default_dtype()
    first = make_basis([0, 1], [4, 16])
    second = make_basis(torch.tensor([[0.], [1.]]), torch.tensor([[4.], [16.]]))
    assert old_dtype == torch.get_default_dtype()
    theta = torch.tensor([0.2, 0.8], dtype=torch.float64)
    torch.testing.assert_close(first(theta), second(theta), atol=0, rtol=0)
    assert first(theta).shape == (2, 2)
    assert first(theta[:, None]).shape == (2, 2)
    assert first(theta[0]).shape == (2,)
    torch.testing.assert_close(first(theta[0]), first(theta)[0], atol=0, rtol=0)


def test_features_bounds_peak_periodicity_and_chordal_value_gradient_parity():
    centers, widths = [value.requires_grad_() for value in geometry()]
    theta = torch.linspace(-math.pi, math.pi, 21, dtype=torch.float64, requires_grad=True)
    direct = circular_features(theta, centers, widths)
    chordal = chordal_features(theta, centers, widths)
    assert direct.shape == (21, 4)
    assert (direct > 0).all() and (direct <= 1).all()
    torch.testing.assert_close(direct, chordal, atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(circular_features(centers, centers, widths).diagonal(),
                               torch.ones(4, dtype=torch.float64), atol=0, rtol=0)
    for shifted in (theta + 2 * math.pi, theta - 4 * math.pi):
        torch.testing.assert_close(circular_features(shifted, centers, widths), direct,
                                   atol=1e-13, rtol=1e-11)
    torch.testing.assert_close(circular_features(theta, centers + 2 * math.pi, widths), direct,
                               atol=1e-13, rtol=1e-11)
    comparison_weight = torch.linspace(-1, 1, direct.numel(), dtype=torch.float64).reshape_as(direct)
    first = torch.autograd.grad((direct * comparison_weight).sum(), (theta, centers, widths))
    second = torch.autograd.grad((chordal * comparison_weight).sum(), (theta, centers, widths))
    for left, right in zip(first, second):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-11)


def test_wrapped_basis_and_functional_kernel_have_identical_geometry():
    basis = make_basis(*geometry(), mode="both")
    theta, _ = example()
    expected = circular_features(theta, basis.centers, basis.concentrations)
    torch.testing.assert_close(basis(theta), expected, atol=0, rtol=0)
    torch.testing.assert_close(basis.chordal_features(theta), expected, atol=1e-14, rtol=1e-12)
    with torch.no_grad():
        basis.bank.raw_kappa.fill_(-50)
    assert (basis.concentrations > 0).all()
    assert torch.isfinite(basis(theta)).all()


def test_narrow_features_underflow_explicitly_without_clipping():
    basis = make_basis([0.], [1000.])
    values = basis(torch.tensor([0., math.pi], dtype=torch.float64))
    torch.testing.assert_close(values[:, 0], torch.tensor([1., 0.], dtype=torch.float64), atol=0, rtol=0)


@pytest.mark.parametrize("centers,widths,mode", [
    ([], [], "fixed"), ([0, 1], [4], "fixed"), ([0], [0], "fixed"),
    ([0], [-1], "both"), ([float("nan")], [4], "fixed"),
    ([0], [float("inf")], "fixed"), ([0], [4], "invalid"),
    ([[0, 1]], [[4, 4]], "fixed"), ([1j], [4], "fixed"),
])
def test_invalid_initialization_rejected(centers, widths, mode):
    with pytest.raises((ValueError, TypeError)):
        make_basis(centers, widths, mode)


@pytest.mark.parametrize("bad_angles", [torch.zeros(3), torch.zeros(2, 2, dtype=torch.float64),
                                        torch.zeros(2, 1, 1, dtype=torch.float64)])
def test_invalid_input_dtype_or_shape_rejected(bad_angles):
    with pytest.raises(ValueError):
        make_basis(*geometry())(bad_angles)


def test_casting_basis_to_float32_requires_explicit_restoration_to_float64():
    basis = make_basis(*geometry()).float()
    theta, _ = example()
    with pytest.raises(ValueError, match="dtype"):
        basis(theta)
    assert basis.double()(theta).dtype == torch.float64


def test_fourier_integer_harmonics_and_periodicity():
    theta, _ = example()
    first = fourier_features(theta, 3)
    second = fourier_features(theta[:, None], [1, 2, 3])
    np.testing.assert_allclose(first.numpy(), np.column_stack([
        np.cos(theta.numpy()), np.cos(2 * theta.numpy()), np.cos(3 * theta.numpy()),
        np.sin(theta.numpy()), np.sin(2 * theta.numpy()), np.sin(3 * theta.numpy())]), atol=1e-14, rtol=1e-13)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    torch.testing.assert_close(first, fourier_features(theta + 2 * math.pi, 3), atol=1e-14, rtol=1e-12)
    assert fourier_features(theta[0], 3).shape == (6,)


@pytest.mark.parametrize("harmonics", [0, -1, True, 1.5, [0, 1], [1, 1], [1, 2.5], []])
def test_invalid_fourier_harmonics_rejected(harmonics):
    with pytest.raises(ValueError):
        fourier_features(torch.zeros(3, dtype=torch.float64), harmonics)


def numpy_augmented_readout(features, target, penalty):
    """Solve via augmented SVD least squares, independent of centered Cholesky."""
    n, k = features.shape
    design = np.column_stack([features, np.ones(n)])
    regularizer = np.column_stack([np.eye(k) * np.sqrt(n * penalty), np.zeros(k)])
    coefficients = np.linalg.lstsq(np.vstack([design, regularizer]),
                                    np.concatenate([target, np.zeros(k)]), rcond=None)[0]
    return coefficients[:-1], coefficients[-1]


@pytest.mark.parametrize("case", ["generic", "constant", "collinear", "underdetermined", "one_sample", "constant_target"])
@pytest.mark.parametrize("column", [False, True])
def test_readout_matches_independent_augmented_numpy_lstsq(case, column):
    theta, y = example()
    features = make_basis(*geometry())(theta)
    if case == "constant":
        features = torch.full_like(features, 3.7)
    elif case == "collinear":
        features = torch.cat([features[:, :2], features[:, :1], 2 * features[:, :1]], dim=1)
    elif case == "underdetermined":
        features, y = features[:2], y[:2]
    elif case == "one_sample":
        features, y = features[:1], y[:1]
    elif case == "constant_target":
        y = torch.full_like(y, 17.)
    expected_weight, expected_bias = numpy_augmented_readout(features.numpy(), y.numpy(), .001)
    target = y[:, None] if column else y
    weight, bias = solve_readout(features, target, .001)
    assert weight.shape == ((4, 1) if column else (4,))
    assert bias.shape == ((1,) if column else ())
    np.testing.assert_allclose(weight.detach().numpy().reshape(-1), expected_weight, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(bias.detach().numpy(), expected_bias, rtol=1e-11, atol=1e-11)
    predictions = predict_readout(features, weight, bias)
    np.testing.assert_allclose(predictions.numpy().reshape(-1), features.numpy() @ expected_weight + expected_bias,
                               rtol=1e-11, atol=1e-11)
    expected_objective = .5 * np.mean((features.numpy() @ expected_weight + expected_bias - y.numpy()) ** 2) + .0005 * np.sum(expected_weight ** 2)
    assert reduced_objective(features, target, .001).item() == pytest.approx(expected_objective, rel=1e-12, abs=1e-14)
    diagnostics = readout_diagnostics(features, target, .001, weight, bias)
    assert diagnostics["relative_backward_error"] < 1e-14
    assert diagnostics["readout_stationarity_l2"] < 1e-12
    assert diagnostics["intercept_stationarity_abs"] < 1e-12
    assert all(math.isfinite(value) for value in diagnostics.values())


def test_intercept_is_not_penalized_and_constant_features_have_zero_weights():
    features = torch.full((13, 5), 3., dtype=torch.float64)
    target = torch.arange(13, dtype=torch.float64) + 100
    for penalty in (.001, 1., 1000.):
        weight, bias = solve_readout(features, target, penalty)
        torch.testing.assert_close(weight, torch.zeros_like(weight), atol=0, rtol=0)
        assert bias.item() == target.mean().item()
        assert reduced_objective(features, target, penalty).item() == .5 * target.var(unbiased=False).item()


@pytest.mark.parametrize("column", [False, True])
def test_reduced_gradient_matches_envelope_and_initial_joint_geometry_gradient(column):
    theta, y = example()
    if column:
        y = y[:, None]
    basis = make_basis(*geometry(), mode="both")
    features = basis(theta)
    weight, bias = solve_readout(features, y, .001)
    reduced = reduced_objective(features, y, .001)
    params = tuple(basis.parameters())
    reduced_grad = torch.autograd.grad(reduced, params, retain_graph=True)
    # Independent envelope calculation freezes the optimal linear readout.
    residual = predict_readout(features, weight.detach(), bias.detach()) - y
    joint = .5 * residual.square().mean() + .0005 * weight.detach().square().sum()
    joint_grad = torch.autograd.grad(joint, params, retain_graph=True)
    for first, second in zip(reduced_grad, joint_grad):
        torch.testing.assert_close(first, second, atol=1e-12, rtol=1e-10)
        # Equal starting ridge readouts imply the same first geometry SGD step.
        torch.testing.assert_close(.01 * first, .01 * second, atol=1e-14, rtol=1e-10)
    feature_grad = torch.autograd.grad(reduced, features)[0]
    analytic = residual.detach().reshape(-1, 1) * weight.detach().reshape(1, -1) / len(y)
    torch.testing.assert_close(feature_grad, analytic, atol=1e-12, rtol=1e-10)


def test_geometry_reduced_gradient_matches_central_finite_differences():
    theta, y = example()
    centers, widths = geometry()
    raw_widths = widths + torch.log(-torch.expm1(-widths))
    packed = torch.cat([centers, raw_widths]).requires_grad_()

    def objective(value):
        return reduced_objective(circular_features(theta, value[:4], F.softplus(value[4:])), y, .001)

    automatic = torch.autograd.grad(objective(packed), packed)[0]
    epsilon = 1e-5
    numerical = []
    for index in range(len(packed)):
        direction = torch.zeros_like(packed)
        direction[index] = epsilon
        numerical.append(((objective(packed.detach() + direction) - objective(packed.detach() - direction)) / (2 * epsilon)).item())
    np.testing.assert_allclose(automatic.numpy(), numerical, rtol=3e-6, atol=2e-9)


def test_readout_autograd_gradcheck_features_targets_and_lambda():
    features = torch.tensor([[.1, .7], [.8, -.2], [.4, .9], [.7, .3]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([1., -.5, .4, .8], dtype=torch.float64, requires_grad=True)
    penalty = torch.tensor(.013, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(reduced_objective, (features, y, penalty), atol=1e-6, rtol=1e-4)


def test_diagnostics_are_detached_and_do_not_replace_a_supplied_bad_readout():
    theta, y = example()
    features = make_basis(*geometry())(theta).requires_grad_()
    weight, bias = solve_readout(features, y, .001)
    before = torch.get_rng_state().clone()
    correct = readout_diagnostics(features, y, .001, weight, bias)
    bad = readout_diagnostics(features, y, .001, weight + 1, bias + .3)
    assert torch.equal(before, torch.get_rng_state())
    assert features.grad is None
    assert all(not isinstance(value, torch.Tensor) for value in correct.values())
    assert correct["relative_backward_error"] < 1e-14
    assert bad["relative_backward_error"] > 1e-4
    assert bad["readout_stationarity_l2"] > .01
    assert bad["intercept_stationarity_abs"] > .01
    with pytest.raises(ValueError, match="both"):
        readout_diagnostics(features, y, .001, weight=weight)
    with pytest.raises(ValueError, match="target shape"):
        readout_diagnostics(features, y[:, None], .001, weight, bias)


@pytest.mark.parametrize("penalty", [0, -1, float("nan"), float("inf"), True, [0.001], 1j])
def test_readout_rejects_invalid_penalty(penalty):
    with pytest.raises(ValueError):
        solve_readout(torch.ones(4, 2, dtype=torch.float64), torch.ones(4, dtype=torch.float64), penalty)


@pytest.mark.parametrize("case", ["float32_features", "float32_y", "target_shape", "empty_rows", "empty_features", "nan_feature", "inf_y"])
def test_readout_rejects_invalid_data(case):
    features = torch.ones(4, 2, dtype=torch.float64)
    y = torch.ones(4, dtype=torch.float64)
    if case == "float32_features":
        features = features.float()
    elif case == "float32_y":
        y = y.float()
    elif case == "target_shape":
        y = y[:, None].repeat(1, 2)
    elif case == "empty_rows":
        features, y = features[:0], y[:0]
    elif case == "empty_features":
        features = features[:, :0]
    elif case == "nan_feature":
        features[0, 0] = float("nan")
    elif case == "inf_y":
        y[0] = float("inf")
    with pytest.raises(ValueError):
        solve_readout(features, y, .001)


def test_finite_data_overflow_does_not_silently_return_invalid_coefficients():
    features = torch.tensor([[1e200], [-1e200]], dtype=torch.float64)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        solve_readout(features, torch.ones(2, dtype=torch.float64), .001)


def test_prediction_rejects_accidental_broadcasting():
    with pytest.raises(ValueError, match="shape"):
        predict_readout(torch.ones(4, 2, dtype=torch.float64), torch.ones(2, dtype=torch.float64),
                        torch.ones(1, dtype=torch.float64))


def test_no_global_rng_consumption_by_entire_numerical_path():
    before = torch.get_rng_state().clone()
    theta, y = example()
    basis = make_basis(*geometry(), mode="both")
    features = basis(theta)
    reduced_objective(features, y, .001).backward()
    weight, bias = solve_readout(features, y, .001)
    predict_readout(features, weight, bias)
    readout_diagnostics(features, y, .001, weight, bias)
    fourier_features(theta, 8)
    assert torch.equal(before, torch.get_rng_state())
