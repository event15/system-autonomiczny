import numpy as np

class Effector:
    def __init__(self, name="Effector", decision_threshold_vd=0.6, reaction_delay_tau=2.5, power_max=1.0):
        self.name = name
        self.decision_threshold_vd = decision_threshold_vd # Vd - decision threshold
        self.reaction_delay_tau = reaction_delay_tau # tau_delay_reakcja - reaction delay
        self.power_max = power_max # Max power consumption when active
        self.reaction = 0.0 # Effector reaction signal (0 or 1)
        self.reaction_time_delay_steps = int(reaction_delay_tau / 0.1) # Delay in steps, assuming dt=0.1

    def process_decision(self, estimation_potential_ve, time_step_index, reaction_history):
        """Processes the estimation potential and triggers reaction if threshold is crossed with delay."""
        decision = False
        last_reaction = 0  # Default to no reaction if history is empty

        if reaction_history.size > 0 and time_step_index > 0:  # Check if reaction history exists and we are past the first step
            last_reaction = reaction_history[time_step_index - 1]  # Get the reaction from the PREVIOUS time step

        if estimation_potential_ve > self.decision_threshold_vd and last_reaction == 0:  # Trigger decision if Ve > Vd and no prior reaction
            decision = True

        if decision:
            delay_index = min(time_step_index + self.reaction_time_delay_steps,
                              len(reaction_history) - 1)  # Cap index to array length
            if delay_index >= 0:
                reaction_history[delay_index] = 1  # Set reaction to 1 after delay - MODIFYING IN-PLACE
                self.reaction = reaction_history[delay_index]  # Update effector reaction - CURRENT reaction value
                return reaction_history  # Return modified reaction_history for consistency

        self.reaction = 0  # No reaction in this step if no decision
        return reaction_history  # Return reaction_history

    def get_reaction(self):
        return self.reaction

    def get_power_consumption(self):
        """Returns power consumption based on reaction state."""
        return self.power_max if self.reaction == 1 else 0.0


# Example usage (can be in simulation.py or separate test file)
if __name__ == '__main__':
    effector = Effector()
    time_points = np.arange(0, 30, 0.1)
    ve_values_example = [] # Example Ve values
    reaction_values = np.zeros_like(time_points)

    for i, t in enumerate(time_points):
        ve_value = 0.7 if 5 < t < 15 else 0.3 # Example Ve values over time
        ve_values_example.append(ve_value)
        reaction_signal = effector.process_decision(ve_value, i, reaction_values) # Pass reaction history
        if np.any(reaction_signal): # If reaction signal is returned (not zeros)
             reaction_values = reaction_signal # Update reaction values

    effector_reaction_output = []
    for r in reaction_values:
        effector_reaction_output.append(effector.get_reaction())

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, ve_values_example, linestyle='--', label='Estimation Potential Ve(t)') # Example Ve
    plt.step(time_points, reaction_values, where='post', label='Effector Reaction (t)') # Step plot for reaction
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Effector - Reaction to Estimation Potential")
    plt.legend()
    plt.grid(True)
    plt.show()
