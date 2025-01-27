import numpy as np

class Corrector:
    def __init__(self, name="Corrector", initial_conductance_g0=0.1, alpha_g=0.05, extinction_r=0.02, extinction_d=0.01):
        self.name = name
        self.conductance_g = initial_conductance_g0  # G
        self.initial_conductance_g0 = initial_conductance_g0
        self.alpha_g = alpha_g  # alpha in ΔG = G0 * α * K
        self.extinction_r = extinction_r # εr - registration extinction rate
        self.extinction_d = extinction_d # εd - deregistration extinction rate
        self.correlation_power_k = 0.0  # K
        self.estimation_potential_ve = 0.0 # Ve
        self.correlation_potential_vk = 0.0 # Vk

    def calculate_correlation_potential(self, vr_sum, vh):
        """Calculates correlation potential Vk = Vr + Vh."""
        self.correlation_potential_vk = vr_sum + vh
        return self.correlation_potential_vk

    def calculate_correlation_power(self):
        """Calculates correlation power K = Vk * G."""
        self.correlation_power_k = self.correlation_potential_vk * self.conductance_g
        return self.correlation_power_k

    def calculate_estimation_potential(self, weight_wt=0.8): # wt from your code
        """Calculates estimation potential Ve = wt * K."""
        self.estimation_potential_ve = weight_wt * self.correlation_power_k
        return self.estimation_potential_ve

    def update_conductance_registration(self, stimulus_value, dt):
        """Updates conductance G during registration (stimulus present). Using formula 8.10."""
        kg = self.calculate_kg_limit()
        gg = self.calculate_gg_limit()
        g_target = gg - (gg - self.initial_conductance_g0) * np.exp(-self.extinction_r * dt) # Formula 8.10 - but for small dt, it is more about change in G
        dg = (g_target - self.conductance_g)
        self.conductance_g += dg #* dt  - Simplified for now, needs refinement based on dt and intended dynamics

    def update_conductance_deregistration(self, dt):
        """Updates conductance G during deregistration (stimulus absent). Using formula 8.18."""
        gp = self.conductance_g # Current G is considered Gp for deregistration
        g_target = self.initial_conductance_g0 + (gp - self.initial_conductance_g0) * np.exp(-self.extinction_d * dt) # Formula 8.18
        dg = (g_target - self.conductance_g)
        self.conductance_g += dg #* dt - Simplified for now, needs refinement based on dt and intended dynamics

    def calculate_kg_limit(self):
        """Calculates the limit value of K (Kg) - formula 8.6."""
        denominator = 1 - self.alpha_g * self.correlation_potential_vk * self.initial_conductance_g0
        if denominator == 0: # Avoid division by zero
            return float('inf')
        return (self.correlation_potential_vk * self.initial_conductance_g0) / denominator

    def calculate_gg_limit(self):
        """Calculates the limit value of G (Gg) - formula 8.7."""
        denominator = 1 - self.alpha_g * self.correlation_potential_vk * self.initial_conductance_g0
        if denominator == 0: # Avoid division by zero
            return float('inf')
        return self.initial_conductance_g0 / denominator

    def get_conductance(self):
        return self.conductance_g

    def get_correlation_power(self):
        return self.correlation_power_k

    def get_estimation_potential(self):
        return self.estimation_potential_ve

    def get_correlation_potential_vk(self):
        return self.correlation_potential_vk


# Example usage (can be in simulation.py or a separate test file)
if __name__ == '__main__':
    corrector = Corrector()
    time_points = np.arange(0, 20, 0.1)
    g_values = []
    k_values = []
    ve_values = []
    vk_values = []
    stimulus_active = False

    for t in time_points:
        stimulus_value = 1.0 if 5 < t < 15 else 0.0
        vh_value = 0.2 # Example Vh

        vk = corrector.calculate_correlation_potential(stimulus_value, vh_value)
        k = corrector.calculate_correlation_power()
        ve = corrector.calculate_estimation_potential()

        if stimulus_value > 0:
            corrector.update_conductance_registration(stimulus_value, 0.1) # dt = 0.1
        else:
            corrector.update_conductance_deregistration(0.1) # dt = 0.1

        g_values.append(corrector.get_conductance())
        k_values.append(corrector.get_correlation_power())
        ve_values.append(corrector.get_estimation_potential())
        vk_values.append(corrector.get_correlation_potential_vk())


    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, g_values, label='Conductance G(t)')
    plt.plot(time_points, k_values, label='Correlation Power K(t)')
    plt.plot(time_points, ve_values, label='Estimation Potential Ve(t)')
    plt.plot(time_points, vk_values, label='Correlation Potential Vk(t)')


    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Corrector Dynamics")
    plt.legend()
    plt.grid(True)
    plt.show()
