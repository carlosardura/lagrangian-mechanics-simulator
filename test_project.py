import numpy as np
import sympy as sp
import pytest
from project import Particle, energies_over_time, run_simulation, x, y

@pytest.fixture
def sho():
    return Particle (
        name = "sho",
        mass = 1.0,
        V_sym = 0.5 * (x**2 + y**2),
        position =[1.0, 0.0],
        velocity = [0.0, 1.0]
    )

@pytest.fixture
def noncentral():
    return Particle (
        name = "noncentral",
        mass = 1.0,
        V_sym = x * y,
        position = [1.0, 1.0],
        velocity = [0.5, 0.0]
    )

@pytest.fixture
def free():
    return Particle (
        name = "free",
        mass = 1.0,
        V_sym = 0,
        position =[0.0, 0.0],
        velocity = [0.0, 0.0]
    )

@pytest.fixture
def particles(sho, noncentral, free):
    return [sho, noncentral, free]

# ---------- main module tests ----------

def test_euler_lagrange_equations(particles):
    # [energy, px, py, Lz]
    assert particles[0].conserved().dtype == bool
    assert particles[0].conserved().tolist() == [True, False, False, True]
    assert particles[1].conserved().tolist() == [True, False, False, False]
    assert particles[2].conserved().tolist() == [True, True, True, True]


def test_run_simulation(particles):
    tspan = (0.0, 50.0)
    h = 0.01

    for p in particles:
        results = run_simulation(p, tspan, h)

        for method in results:
            r = results[method]
            assert np.all(np.isfinite(r["pos"]))
            assert np.all(np.isfinite(r["vel"]))
            assert np.all(np.isfinite(r["E"])) 

            for k in r:
                assert len(r[k]) > 0
                n = len(r[k][:,0]) if r[k].ndim > 1 else len(r[k])
                assert np.array_equal(n, int((tspan[1]-tspan[0])/h) + 1)

        assert np.all(r["T"] >= 0)
        assert np.array_equal(r["T"] + r["V"], r["E"])


def test_energies_over_time(particles):
    pos = np.array([[1.0, 0.0], [0.0, 1.0]])
    vel = np.array([[0.5, 0.0], [0.0, 0.5]])

    for p in particles:
        # energies = [T, V, E]
        energies = energies_over_time(p, pos, vel)
        assert np.all(energies[0] >= 0)
        assert np.all(energies[0] + energies[1] == energies[2])
        for i in energies:
            assert len(i) == len(pos)


def test_integrators():
    pass