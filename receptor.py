import numpy as np

class Receptor:
    def __init__(self, name="Receptor", noise_amplitude=0.05):
        self.name = name
        self.noise_amplitude = noise_amplitude
        self.vr = 0.0  # Receptor Potential

    def process_stimulus(self, stimulus_value):
        """Processes the stimulus and generates receptor potential (Vr)."""
        noise = self.noise_amplitude * np.random.normal()
        self.vr = stimulus_value + noise
        return self.vr

    def get_potential(self):
        return self.vr

# Example of a group of receptors (can be in simulation.py or a separate test file)
if __name__ == '__main__':
    receptor1 = Receptor(name="TemperatureReceptor")
    receptor2 = Receptor(name="LightReceptor")

    time_points = np.arange(0, 10, 0.1)
    vr1_values = []
    vr2_values = []

    for t in time_points:
        stimulus1 = 0.5 if 2 < t < 7 else 0  # Example stimulus for Receptor 1
        stimulus2 = 0.8 if 5 < t < 9 else 0  # Example stimulus for Receptor 2

        vr1 = receptor1.process_stimulus(stimulus1)
        vr2 = receptor2.process_stimulus(stimulus2)

        vr1_values.append(vr1)
        vr2_values.append(vr2)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(time_points, vr1_values, label=f'{receptor1.name} (Vr)')
    plt.plot(time_points, vr2_values, label=f'{receptor2.name} (Vr)')
    plt.xlabel("Time")
    plt.ylabel("Receptor Potential (Vr)")
    plt.title("Receptor Potentials over Time")
    plt.legend()
    plt.grid(True)
    plt.show()