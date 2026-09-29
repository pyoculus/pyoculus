"""Field configuration must represent the same physics before and after JIT."""

from copy import deepcopy
import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("jax")
from pyoculus.fields import AnalyticCylindricalBfield as Field

POINT = np.array([6.7, 0.23, 0.6])


def test_implicit_and_explicit_centers(perturbation):
    """An omitted center follows the equilibrium; an explicit coil stays put."""
    explicit = dict(perturbation, R=7.0, Z=-1.0)
    field = Field(6, 0.2, 1.16, 1.2, [perturbation, explicit])
    assert (field.perturbations_args[0]["R"], field.perturbations_args[0]["Z"]) == (
        6,
        0.2,
    )
    assert (field.perturbations_args[1]["R"], field.perturbations_args[1]["Z"]) == (
        7,
        -1,
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Constructor fills R/Z in caller-owned dictionaries",
)
def test_construction_preserves_input(perturbation):
    """A reusable physical specification should not acquire hidden centers."""
    original = deepcopy(perturbation)
    Field(6, 0, 1.16, 1.2, [perturbation])
    assert perturbation == original


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Reused dictionaries keep the first equilibrium's implicit center",
)
def test_reused_specification_tracks_each_equilibrium(perturbation):
    """Translating an equilibrium must translate its implicitly centered mode."""
    specs = [perturbation]
    Field(6, 0, 1.16, 1.2, specs)
    second = Field(7, 1, 1.16, 1.2, specs)
    assert (second.perturbations_args[0]["R"], second.perturbations_args[0]["Z"]) == (
        7,
        1,
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Fields alias the input list; amplitudes and warmed JIT disagree",
)
def test_shared_input_does_not_couple_fields(perturbation):
    """Changing one field must not alter another field's configured strength."""
    specs = [perturbation]
    first = Field(6, 0, 1.16, 1.2, specs)
    second = Field(6, 0, 1.16, 1.2, specs)
    before = second.B(POINT)
    first.set_amplitude(0, 0.2)
    assert second.amplitudes == [0.1]
    assert_allclose(second.B(POINT), before, rtol=0, atol=0)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Default perturbation list is shared between instances",
)
def test_default_fields_are_independent(perturbation):
    """Adding a mode to one initially unperturbed field must not affect another."""
    first = Field(6, 0, 1.16, 1.2)
    second = Field(6, 0, 1.16, 1.2)
    try:
        first.add_perturbation(dict(perturbation, R=6, Z=0))
        assert second.amplitudes == []
    finally:
        # Restore the production default list even when the regression fails.
        first.remove_perturbation()


def test_supported_updates_preserve_superposition(perturbation):
    """B, A and dB are linear in mode amplitude, including after compilation.

    Compare differences from the equilibrium; 1e-11 allows floating-point
    subtraction while remaining far below the finite perturbation signal.
    """
    field = Field(6, 0, 1.16, 1.2, [deepcopy(perturbation)])
    equilibrium = Field(6, 0, 1.16, 1.2, [])

    def values(f):
        return np.concatenate([f.B(POINT), f.A(POINT), f.dBdX(POINT)[1].ravel()])

    base = values(equilibrium)
    delta = values(field) - base
    field.set_amplitude(0, 0.2)
    assert_allclose(values(field) - base, 2 * delta, atol=1e-11)
    field.amplitudes = [0.0]
    assert_allclose(values(field), base, atol=1e-11)
    field.set_perturbation(0, dict(perturbation, R=6, Z=0))
    assert_allclose(values(field) - base, delta, atol=1e-11)
    field.remove_perturbation()
    assert_allclose(values(field), base, atol=1e-11)
    field.add_perturbation(dict(perturbation, R=6, Z=0))
    assert_allclose(values(field) - base, delta, atol=1e-11)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="_jitted_perturbations invalidation misspells cached _jited_perturbations",
)
def test_individual_modes_refresh_after_amplitude_change(perturbation):
    """A warmed individual mode must still sum to the new total perturbation."""
    f = Field(6, 0, 1.16, 1.2, [perturbation])
    f.perturbations[0](POINT)
    f.set_amplitude(0, 0.2)
    assert_allclose(
        f.perturbations[0](POINT), f.B(POINT) - f.B_equilibrium(POINT), atol=1e-11
    )
