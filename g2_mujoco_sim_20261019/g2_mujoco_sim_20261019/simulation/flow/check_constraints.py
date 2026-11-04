"""Cross-check the scalar-multiplier QP against SciPy's independent optimizer."""
import numpy as np
from scipy.optimize import minimize
from constraints import solve_box_clf


def main():
    rng = np.random.default_rng(7)
    for _ in range(12):
        f, a = rng.normal(size=(2, 8))
        lower, upper = np.full(8, -.3), np.full(8, .4)
        lower[0] = upper[0] = 0.  # Fixed endpoint coordinate.
        b, rho = -.7, 30.
        v, slack = solve_box_clf(f, a, b, lower, upper, rho)
        result = minimize(lambda z: .5*np.square(z[:-1]-f).sum()+rho/2*z[-1]**2,
            np.r_[np.clip(f, lower, upper), max(0., a@np.clip(f, lower, upper)-b)],
            method='SLSQP', bounds=list(zip(lower, upper))+[(0., None)],
            constraints=[dict(type='ineq', fun=lambda z: b+z[-1]-a@z[:-1])],
            options=dict(ftol=1e-11, maxiter=500))
        assert result.success, result.message
        np.testing.assert_allclose(v, result.x[:-1], atol=2e-5)
        np.testing.assert_allclose(slack, result.x[-1], atol=2e-5)
        assert a@v <= b+slack+1e-9
        assert np.all(v >= lower) and np.all(v <= upper)
    print('PASS: 12 QPs agree with independent SLSQP solutions')


if __name__ == '__main__':
    main()
