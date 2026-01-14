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
        velocity = [0.0, 1.0],
    )

@pytest.fixture
def coupled():
    return Particle (
        name = "coupled",
        mass = 1.0,
        V_sym = x**2 + y**2 + x*y,
        position = [1/np.sqrt(2), -1/np.sqrt(2)],
        velocity = [-1/np.sqrt(2), 1/np.sqrt(2)]
    )

@pytest.fixture
def free():
    return Particle (
        name = "free",
        mass = 1.0,
        V_sym = 0,
        position =[0.0, -1.0],
        velocity = [1.0, 0.0]
    )

@pytest.fixture
def particles(sho, coupled, free):
    return [sho, coupled, free]

# ---------- main module tests ----------

def test_euler_lagrange_equations(particles):
    acc0 = particles[0].acceleration(particles[0].position, particles[0].velocity)
    assert np.array_equal(acc0, -particles[0].position)
    acc1 = particles[1].acceleration(particles[1].position, particles[1].velocity)
    assert np.array_equal(acc1, -2*particles[1].position-particles[1].position[::-1])
    acc2 = particles[2].acceleration(particles[2].position, particles[2].velocity)
    assert np.array_equal(acc2, [0.0, 0.0])
    
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


def test_integrators(particles):
    tspan = (0.0, 200.0)
    h = 0.01

    for p in particles:
        results = run_simulation(p, tspan, h)

        for method in results:
            E = results[method]["E"]
            rel_error = np.abs(E - E[0]) / max(abs(E[0]), 1e-12)
            if method == "Euler-Cromer":
                for i in range(int(tspan[1]-tspan[0])):
                    assert abs(rel_error[i]) < 1.01e-2
            elif method == "Runge-Kutta 4":
                for i in range(int(tspan[1]-tspan[0])):
                    assert rel_error[i] < 1e-15 + 1.39e-12 * (tspan[0] + h*i)
            elif method == "Velocity Verlet":
                for i in range(int(tspan[1]-tspan[0])):
                    assert abs(rel_error[i]) < 1.25e-5