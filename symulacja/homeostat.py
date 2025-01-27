import numpy as np

class Homeostat:
    def __init__(self, name="Homeostat", alpha_homeostat=0.05, equilibrium_wr=0.5, decision_threshold_vd=0.6):
        self.name = name
        self.alpha_homeostat = alpha_homeostat  # alpha from your code for homeostat
        self.equilibrium_wr = equilibrium_wr  # wr - equilibrium coefficient
        self.decision_threshold_vd = decision_threshold_vd  # Vd - decision threshold
        self.reflection_potential_vh = 0.0  # Vh - Reflection Potential
        self.perturbation_potential_vp = 0.0  # Vp - Perturbation Potential

    def update_reflection_potential(self, estimation_potential_ve, dt):
        """Updates reflection potential Vh based on estimation potential Ve and homeostatic parameters."""
        dVh = -self.alpha_homeostat * (
                    estimation_potential_ve - self.equilibrium_wr * self.decision_threshold_vd) * dt  # Formula from your original code
        self.reflection_potential_vh += dVh
        return self.reflection_potential_vh

    def get_reflection_potential(self):
        return self.reflection_potential_vh

    def set_perturbation_potential(self,
                                   stimulus_sum):  # Vp is set directly from stimulus for now, as per your original code
        """Sets perturbation potential Vp. For now directly from stimulus sum."""
        self.perturbation_potential_vp = stimulus_sum  # Direct mapping as in original code for Vp = suma_bodzcow
        return self.perturbation_potential_vp

    def get_perturbation_potential(self):
        return self.perturbation_potential_vp


# Example usage (can be in simulation.py or separate test file)
if __name__ == '__main__':
    homeostat = Homeostat()
    time_points = np.arange(0, 20, 0.1)
    vh_values = []
    ve_values_example = []  # Example Ve values

    for t in time_points:
        ve_value = 0.7 if 5 < t < 15 else 0.3  # Example Ve values over time
        ve_values_example.append(ve_value)
        homeostat.update_reflection_potential(ve_value, 0.1)  # dt = 0.1
        vh_values.append(homeostat.get_reflection_potential())

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(time_points, vh_values, label='Reflection Potential Vh(t)')
    plt.plot(time_points, ve_values_example, linestyle='--', label='Example Estimation Potential Ve(t)')  # Example Ve
    plt.xlabel("Time")
    plt.ylabel("Potential Value")
    plt.title("Homeostat - Reflection Potential Dynamics")
    plt.legend()
    plt.grid(True)
    plt.show()
