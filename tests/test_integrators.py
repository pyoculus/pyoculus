import numpy as np
from pyoculus.integrators import ScipyODEIntegrator

class TestODEIntegrator:
    def test_exponential_growth(self):
        # Define the differential equation
        def dydt(t, y):
            return y  # This corresponds to k=1

        # Initial condition
        yi = 1.0
        # Time interval for integration
        ti, tf = 0, 1
        # Create an instance of the ODEIntegrator
        integrator = ScipyODEIntegrator(ode=dydt)
        
        # Solve the differential equation
        integrator.set_initial_value(ti, yi)
        yf = integrator.integrate(tf)
        
        # Expected solution at t = tf
        expected_solution = yi * np.exp(tf)

        # Check if the numerical solution at tf is close to the expected solution
        assert np.isclose(yf[0], expected_solution, atol=1e-3), "Numerical solution does not match expected solution within tolerance."


def test_harmonic_oscillator_refinement():
    """q'=p, p'=-q: tighter ODE tolerances reduce phase and energy errors."""

    from numpy.testing import assert_allclose

    def solve(tol):
        integrator = ScipyODEIntegrator(
            ode=lambda t, y: [y[1], -y[0]], rtol=tol, atol=tol * 0.01
        )
        integrator.set_initial_value(0, [1.0, 0.0])
        forward = integrator.integrate(20.0).copy()
        back = integrator.integrate(0.0)
        return forward, back

    loose, _ = solve(1e-5)
    tight, back = solve(1e-10)
    exact = np.array([np.cos(20), -np.sin(20)])
    assert np.linalg.norm(tight - exact) < np.linalg.norm(loose - exact) / 100
    assert_allclose(tight, exact, atol=1e-8)
    assert_allclose(back, [1, 0], atol=1e-8)
