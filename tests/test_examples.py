"""Examples are executable scientific explanations, not untested snippets."""

import importlib.util
from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.mark.parametrize("name", ["greene_flux_surfaces", "qfm_surfaces"])
def test_physics_example(name, tmp_path):
    """Each example computes its physical baseline and renders headlessly.

    The perturbed QFM surface must depart measurably from the integrable one;
    merely producing finite arrays would miss a dropped perturbation.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    data = example.compute()
    if name == "greene_flux_surfaces":
        target, fps = data
        errors = [abs(fp.n / fp.m - target) for fp in fps]
        assert np.all(np.diff(errors) < 0)
        assert [(fp.n, fp.m) for fp in fps] == [(1, 2), (2, 3), (3, 5), (5, 8)]
    else:
        _, _, surfaces = data
        assert np.max(np.abs(surfaces[0.002][0] - 0.5)) > 1e-4
        assert_allclose(surfaces[0.0][0], 0.5, atol=1e-10)
    fig = example.plot(*data)
    try:
        fig.savefig(tmp_path / f"{name}.png")
        assert (tmp_path / f"{name}.png").stat().st_size > 1000
    finally:
        plt.close(fig)
