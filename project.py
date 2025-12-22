import csv

def main():
    particle = input("Select a particle: ")

def select_particle(particle):
    """
    Reads "particles.csv" and allows the user to select a particle.
    Returns a dictionary with the mass of the selected particle.
    """
    pass
    
def select_potential():
    """
    Allows the user to select a potential function for the simulation.
    Returns a callable potential object.
    """
    pass

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
        self.position = [0.0, 0.0]
        self.velocity = [0.0, 0.0]

if __name__ == "__main__":
    main()