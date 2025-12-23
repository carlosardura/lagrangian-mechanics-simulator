import numpy as np
import csv

def main():
    p_name = input("Select a particle: ").strip().lower()
    p_data = select_particle(p_name)
    r0 = np.array([1.0, 0.0])
    v0 = np.array([0.0, 1.0])

    particle = Particle(
        name = p_data["name"],
        mass = p_data["mass"],
        position = r0,
        velocity = v0,
    )

def select_particle(p_name):
    """
    Reads "particles.csv" and allows the user to select a particle.
    Returns a dictionary with the mass of the selected particle.
    """
    with open("particles.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["name"] == p_name:
                return {
                    "name": row["name"],
                    "mass": float(row["mass"]),
                }
            else:
                raise ValueError("Particle not found")
    
def select_potential():
    """
    Allows the user to select a potential function for the simulation.
    Returns a callable potential, with k, G and M equal to 1 for simplicity.
    """
    def V_harmonic(position, k=1.0):
        """
        2D simple harmonic oscillator.
        k: spring constant
        """
        x, y = position
        return 0.5 * k * (x**2 + y**2)
    
    def V_anisotropic(position, kx=1.0, ky=2.0):
        """
        2D anisotropic harmonic oscillator.
        kx, ky: spring constants in x and y
        """
        x, y = position
        return 0.5 * (kx * x ** 2 + ky * y ** 2)
    
    def V_kepler(position, G=1.0, M=1.0):
        """
        Keplerian potential.
        G: gravitational constant
        M: central mass
        """
        r = np.linalg.norm(position)
        if r == 0:
            raise ValueError("Singularity at r = 0")
        return -G * M / r
    
    def V_noncentral(position, alpha=1.0):
        """
        Simple noncentral potential.
        alpha: coupling constant
        """
        x, y = position
        return alpha * x * y
    
    potentials = {
        "1. Simple Harmonic Oscillator": V_harmonic,
        "2. Anisotropic Harmonic Oscillator": V_anisotropic,
        "3. Keplerian": V_kepler,
        "4. Noncentral": V_noncentral,
    }

    print("Available potentials:")
    for name in potentials:
        print(name)
        
    while True:
        index = input("Select a potential (1-4): ").lstrip()
        if not index:
            print("Invalid selection. Try again.")
            continue
        for name, func in potentials.items():
            if index[0] == name[0]:
                return func
        print("Invalid selection. Try again.")

def run_simulation():
    """
    Given a specified integrator, simulates the particle's motion over time.
    """
    pass

class Particle:
    """
    Represents a particle in 2D space with several physical properties.
    """
    def __init__(self, name, mass, position=None, velocity=None):
        self.name = name
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

if __name__ == "__main__":
    main()