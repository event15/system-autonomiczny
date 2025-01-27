#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from receptor import Receptor
from corrector import Corrector
from homeostat import Homeostat
from effector import Effector
import subprocess

# --- Global Simulation Parameters ---
SIMULATION_TIME_MAX = 80.0  # Total simulation time [s]
TIME_STEP_DT = 0.1  # Time step [s]

# --- System Parameters ---
HOMEOSAT_ALPHA = 0.05
HOMEOSAT_EQUILIBRIUM_WR = 0.5
DECISION_THRESHOLD_VD = 0.1
REGISTRATION_BETA = 0.08
DEREGISTRATION_BETA = 0.03
REGISTRATION_DELAY_TAU = 2.0
REACTION_DELAY_TAU = 2.5
ESTIMATION_WEIGHT_WT = 0.8
RECEPTOR_NOISE_AMPLITUDE = 0.05
INITIAL_CONDUCTANCE_G0 = 0.1
ALPHA_G_CONDUCTANCE = 0.05
EXTINCTION_REGISTRATION_R = 0.02
EXTINCTION_DEREGISTRATION_D = 0.01
EFFECTOR_POWER_MAX = 1.0


def create_stimuli(time_vector):
    """Defines and sums stimuli over time."""
    stimuli_list = [
        {"start": 10.0, "stop": 30.0, "offset": 2.0, "amplitude_sin": 0.0, "frequency_sin": 0.0},
        # Bardzo silny bodziec krokowy

        {"start": 10.0, "stop": 20.0, "offset": 0.7, "amplitude_sin": 0.0, "frequency_sin": 0.0}, # Prosty bodziec krokowy
        {"start": 10.0, "stop": 12.0, "offset": 0.5, "amplitude_sin": 0.0, "frequency_sin": 0.0}, # Krótszy bodziec 1
        {"start": 20.0, "stop": 22.0, "offset": 0.5, "amplitude_sin": 0.0, "frequency_sin": 0.0}, # Krótszy bodziec 2
        {"start": 30.0, "stop": 32.0, "offset": 0.5, "amplitude_sin": 0.0, "frequency_sin": 0.0}, # Krótszy bodziec 3
        {"start": 15.0, "stop": 15.1, "offset": 2.0, "amplitude_sin": 0.0, "frequency_sin": 0.0}, # Impuls

    ]
    summed_stimulus = np.zeros_like(time_vector)
    for stimulus in stimuli_list:
        start_index = int(stimulus["start"] / TIME_STEP_DT)
        stop_index = int(stimulus["stop"] / TIME_STEP_DT)
        start_index = max(0, min(start_index, len(time_vector)))
        stop_index = max(0, min(stop_index, len(time_vector)))
        time_segment = time_vector[start_index:stop_index]
        summed_stimulus[start_index:stop_index] += (
                stimulus["offset"] + stimulus["amplitude_sin"] *
                np.sin(stimulus["frequency_sin"] * time_segment)
        )
    return summed_stimulus


def run_simulation():
    """Runs the simulation for the autonomous system."""
    time_vector = np.arange(0, SIMULATION_TIME_MAX, TIME_STEP_DT)
    summed_stimulus = create_stimuli(time_vector)
    num_time_steps = len(time_vector)

    # Initialize Organs
    receptor = Receptor(noise_amplitude=RECEPTOR_NOISE_AMPLITUDE)
    corrector = Corrector(initial_conductance_g0=INITIAL_CONDUCTANCE_G0, alpha_g=ALPHA_G_CONDUCTANCE,
                          extinction_r=EXTINCTION_REGISTRATION_R, extinction_d=EXTINCTION_DEREGISTRATION_D)
    homeostat = Homeostat(alpha_homeostat=HOMEOSAT_ALPHA, equilibrium_wr=HOMEOSAT_EQUILIBRIUM_WR,
                          decision_threshold_vd=DECISION_THRESHOLD_VD)
    effector = Effector(decision_threshold_vd=DECISION_THRESHOLD_VD, reaction_delay_tau=REACTION_DELAY_TAU,
                        power_max=EFFECTOR_POWER_MAX)

    # Initialize data arrays for plotting
    vr_values = np.zeros(num_time_steps)
    vh_values = np.zeros(num_time_steps)
    ve_values = np.zeros(num_time_steps)
    g_values = np.zeros(num_time_steps)
    k_values = np.zeros(num_time_steps)
    vk_values = np.zeros(num_time_steps)  # Added vk_values initialization here
    reaction_values = np.zeros(num_time_steps)
    vp_values = np.zeros(num_time_steps)
    consciousness_values = np.zeros(num_time_steps)  # Swiadomosc
    thinking_values = np.zeros(num_time_steps)  # Myslenie

    decision_times = []
    reaction_times = []
    reaction_history = []  # To track reaction in effector

    registration_delay_steps = int(REGISTRATION_DELAY_TAU / TIME_STEP_DT)

    for i in range(num_time_steps):
        # 1. Receptor - Process Stimulus
        vr = receptor.process_stimulus(summed_stimulus[i])
        vr_values[i] = vr

        # 2. Homeostat - Set Perturbation Potential (Vp), Update Reflection Potential (Vh)
        vp = homeostat.set_perturbation_potential(summed_stimulus[i])  # Vp = stimulus sum as per original code
        vp_values[i] = vp

        # 3. Corrector - Calculate Potentials and Power, Update Conductance
        vr_delayed_input = vr_values[
            max(0, i - registration_delay_steps)] if i > registration_delay_steps else 0  # Registration delay
        vk = corrector.calculate_correlation_potential(vr_delayed_input, homeostat.get_reflection_potential())
        k = corrector.calculate_correlation_power()
        ve = corrector.calculate_estimation_potential(weight_wt=ESTIMATION_WEIGHT_WT)  # wt as parameter
        corrector.correlation_potential_vk = vk  # Update Vk in corrector for G limits calculations

        vh = homeostat.update_reflection_potential(ve, TIME_STEP_DT)  # Update Vh based on Ve

        vk_values[i] = vk
        k_values[i] = k
        ve_values[i] = ve
        vh_values[i] = vh
        g_values[i] = corrector.get_conductance()

        if summed_stimulus[i] > 0:  # Stimulus present - Registration
            corrector.update_conductance_registration(summed_stimulus[i], TIME_STEP_DT)
        else:  # Stimulus absent - Deregistration
            corrector.update_conductance_deregistration(TIME_STEP_DT)

        # 4. Effector - Process Decision and Reaction
        reaction_signal_effector = effector.process_decision(ve, i, reaction_values)  # Pass reaction history
        if np.any(reaction_signal_effector):  # If reaction signal is returned (not zeros)
            reaction_values = reaction_signal_effector  # Update reaction values

        # 5. Consciousness and Thinking (from your original code)
        consciousness_values[i] = k * (vp - vh)  # Swiadomosc = K*(Vp-Vh)
        thinking_values[i] = thinking_values[
                                 i - 1] + k * vh * TIME_STEP_DT if i > 0 else 0  # Myslenie = integral of K*Vh*dt

        # 6. Record Decision and Reaction Times (simplified - using effector's reaction)
        if reaction_values[i] == 1 and (i == 0 or reaction_values[i - 1] == 0):  # Reaction starts at this step
            reaction_times.append(time_vector[i])
            decision_times.append(time_vector[i] - REACTION_DELAY_TAU)  # Estimate decision time by subtracting delay

    return (time_vector, summed_stimulus, vr_values, vh_values, ve_values, g_values, k_values, vk_values,
            reaction_values, decision_times, reaction_times, vp_values, consciousness_values, thinking_values)


def plot_simulation_results(time_vector, summed_stimulus, vr_values, vh_values, ve_values, g_values, k_values,
                            vk_values,
                            reaction_values, decision_times, reaction_times, vp_values, consciousness_values,
                            thinking_values):
    """Generates plots for the simulation results."""
    fig = plt.figure(figsize=(16, 27))
    # ... (rest of your plotting code from previous version - adapt colors dictionary if needed) ...
    # --- Colors Dictionary (consistent with previous) ---
    colors = {
        'bodziec': '#8A2BE2',
        'vr': '#4169E1',
        'vh': '#228B22',
        've': '#8B4513',
        'G': '#FF8C00',
        'K': '#DC143C',
        'vk': '#4B0082',
        'reakcja': '#B22222',
        'vp': '#6A5ACD',
        'swiadomosc': '#FF4500',
        'myslenie': '#4B0082'
    }
    xticks = np.arange(0, SIMULATION_TIME_MAX + 1, 20)
    xlabel = 'Time [s]'

    # Panel 1: Information Pathway
    ax1 = fig.add_subplot(9, 1, 1)
    ax1.plot(time_vector, summed_stimulus, label='Stimulus (Sum)', color=colors['bodziec'], lw=1.5)
    ax1.plot(time_vector, vr_values, label='Vr (Receptor)', color=colors['vr'], alpha=0.8, linestyle='--')
    ax1.plot(time_vector, vh_values, label='Vh (Homeostat)', color=colors['vh'], alpha=0.8, linestyle=':')
    ax1.axhline(DECISION_THRESHOLD_VD, color='black', linestyle='-.',
                label=f'Decision Threshold (Vd={DECISION_THRESHOLD_VD})')
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([])
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax1.set_title('A) Information Pathway: Inputs and Potentials', fontsize=10, pad=12)

    # Panel 2: Registration/Deregistration
    ax2 = fig.add_subplot(9, 1, 2)
    ax2.plot(time_vector, g_values, label='Conductance G(t)', color=colors['G'], lw=1.5)
    ax2.plot(time_vector, k_values, label='Power K(t)', color=colors['K'], alpha=0.8)
    ax2.fill_between(time_vector, 0, k_values, where=(k_values > 0), color='#FFB6C1', alpha=0.2)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([])
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_title('B) Registration/Deregistration', fontsize=10, pad=12)

    # Panel 3: Estimation and Decisions
    ax3 = fig.add_subplot(9, 1, 3)
    ax3.plot(time_vector, ve_values, label='Ve (Estimation)', color=colors['ve'], lw=1.5)
    ax3.axhline(DECISION_THRESHOLD_VD, color='black', linestyle='-.')

    for i, cd in enumerate(decision_times):
        y_offset = DECISION_THRESHOLD_VD + 0.3 + (i % 2) * 0.2
        x_offset = cd + 1.5 + (i * 0.7)
        ax3.annotate(f'td={cd:.1f}s',
                     xy=(cd, DECISION_THRESHOLD_VD),
                     xytext=(x_offset, y_offset),
                     fontsize=7,
                     arrowprops=dict(arrowstyle="->", lw=0.7, relpos=(0, 0.2)),
                     textcoords='data')

    ax3.set_xticks(xticks)
    ax3.set_xticklabels([])
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_title('C) Estimation and Decisions', fontsize=10, pad=15)

    # Panel 4: Corrector-Effector Relation
    ax4 = fig.add_subplot(9, 1, 4)
    ax4.plot(time_vector, k_values, label='Power K(t)', color=colors['K'], alpha=0.8)
    ax4.axhline(DECISION_THRESHOLD_VD / ESTIMATION_WEIGHT_WT, color='#FF4500', linestyle='--',
                label=f'Threshold K: {DECISION_THRESHOLD_VD / ESTIMATION_WEIGHT_WT:.2f}')
    ax4.set_xticks(xticks)
    ax4.set_xticklabels([])
    ax4.legend(loc='upper right', fontsize=8)
    ax4.set_title('D) Corrector-Effector Relation', fontsize=10, pad=12)

    # Panel 5: Correlation Potentials
    ax5 = fig.add_subplot(9, 1, 5)
    ax5.plot(time_vector, vr_values, label='Vr (Registration)', color=colors['vr'], alpha=0.8)
    ax5.plot(time_vector, vh_values, label='Vh (Reflection)', color=colors['vh'], alpha=0.8)
    ax5.plot(time_vector, vk_values, label='Vk = Vr + Vh', color=colors['vk'], lw=1.5)
    ax5.set_xticks(xticks)
    ax5.set_xticklabels([])
    ax5.legend(loc='upper right', fontsize=8)
    ax5.set_title('E) Correlation Potentials', fontsize=10, pad=12)

    # Panel 6: Energy and Conductance
    ax6 = fig.add_subplot(9, 1, 6)
    ax6.plot(time_vector, k_values * TIME_STEP_DT, label='Correlation Energy', color='#FF1493', lw=1.2)
    ax6.bar(time_vector[::10], g_values[::10], width=0.8, color='#FFD700', alpha=0.6, label='Conductance G(t)')
    ax6.set_xticks(xticks)
    ax6.set_xticklabels([])
    ax6.legend(loc='upper right', fontsize=8)
    ax6.set_title('F) Energy and Conductance', fontsize=10, pad=12)

    # Panel 7: Reactions with Delay
    ax7 = fig.add_subplot(9, 1, 7)
    ax7.step(time_vector, reaction_values, label='Reaction', color=colors['reakcja'], where='post')
    ax7.set_ylim(-0.1, 1.5)
    ax7.set_yticks([0, 1])
    ax7.set_xticks(xticks)
    ax7.set_xticklabels([])
    ax7.legend(loc='upper right', fontsize=8)
    ax7.set_title(f'G) Reactions with Delay ∆td={REACTION_DELAY_TAU}s', fontsize=10, pad=12)

    # Panel 8: Decisions vs Reactions
    ax8 = fig.add_subplot(9, 1, 8)
    ax8.vlines(decision_times, ymin=0, ymax=1, colors='green', label='Decision Time', lw=1.2)
    ax8.vlines(reaction_times, ymin=0, ymax=1, colors='red', label='Reaction Time', lw=1.2)

    for d, r in zip(decision_times, reaction_times):
        delta = r - d
        ax8.annotate(f'Δt={delta:.1f}s', xy=(d, 0.5),
                     xytext=(d + 1, 0.7), fontsize=7,
                     arrowprops=dict(arrowstyle="->", lw=0.7))

    ax8.set_xticks(xticks)
    ax8.tick_params(axis='x', rotation=45, labelsize=8)
    ax8.set_xlabel(xlabel, fontsize=9)
    ax8.set_title('H) Decisions vs Reactions', fontsize=10, pad=12)
    ax8.set_yticks([])
    ax8.legend(loc='upper right', fontsize=8)

    # Panel 9: Consciousness and Thinking
    ax9 = fig.add_subplot(9, 1, 9)
    ax9.plot(time_vector, vh_values, label='Vh (Homeostat)', color='#228B22', alpha=0.8)
    ax9.plot(time_vector, vp_values, label='Vp (Perturbation)', color='#6A5ACD', alpha=0.8)
    ax9.fill_between(time_vector, vh_values, vp_values, where=(vp_values > vh_values), facecolor='#90EE90', alpha=0.3,
                     label='Positive Flow (+)')
    ax9.fill_between(time_vector, vh_values, vp_values, where=(vp_values <= vh_values), facecolor='#FFB6C1', alpha=0.3,
                     label='Negative Flow (-)')

    ax9b = ax9.twinx()
    ax9b.plot(time_vector, consciousness_values, label='Consciousness (K·ΔV)', color='#FF4500', lw=1.2)
    ax9b.plot(time_vector, thinking_values, label='Thinking (∫K·Vh dt)', color='#4B0082', linestyle='--')

    ax9.set_xlabel('Time [s]', fontsize=9)
    ax9.set_ylabel('Potential', fontsize=8)
    ax9b.set_ylabel('Flows', fontsize=8)
    ax9.set_title('I) Consciousness and Thinking: Potential and Flow Dynamics', fontsize=10, pad=12)

    lines, labels = ax9.get_legend_handles_labels()
    lines2, labels2 = ax9b.get_legend_handles_labels()
    ax9.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=7)

    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(hspace=0.5)
    plt.show()


def main():
    """Main function to run simulation and plot results."""
    try:
        simulation_results = run_simulation()
        plot_simulation_results(*simulation_results)

        # --- ZAPIS DO CSV (jak w poprzednim kroku) ---
        time_vector, summed_stimulus, vr_values, vh_values, ve_values, g_values, k_values, vk_values, reaction_values, decision_times, reaction_times, vp_values, consciousness_values, thinking_values = simulation_results
        df_results = pd.DataFrame({
            't': time_vector,
            'stimulus': summed_stimulus,
            'Vr': vr_values,
            'Vh': vh_values,
            'Ve': ve_values,
            'G': g_values,
            'K': k_values,
            'Vk': vk_values,
            'reakcja': reaction_values,
            'Vp': vp_values,
            'swiadomosc': consciousness_values,
            'myslenie': thinking_values
        })
        # --- POPRAWIONE USTAWIENIE KOLUMNY 'decyzja' ---
        decision_array = np.zeros_like(time_vector)
        for decision_time in decision_times:
            decision_index = np.argmin(np.abs(time_vector - decision_time))  # Znajdź najbliższy indeks czasu decyzji
            decision_array[decision_index] = 1
        df_results['decyzja'] = decision_array
        # --- KONIEC POPRAWKI KOLUMNY 'decyzja' ---
        csv_filename = "simulation_results.csv"
        df_results.to_csv(csv_filename, index=False)
        print(f"Simulation results saved to {csv_filename}")
        # --- KONIEC ZAPISU CSV ---

        # --- URUCHAMIANIE ANALIZA.PY ---
        print("Running analiza.py...")
        subprocess.run(['python', 'symulacja/analiza.py'])  # Uruchomienie analizy w osobnym procesie
        print("Analiza completed.")
        # --- KONIEC URUCHAMIANIA ANALIZA.PY ---


    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C)")
        sys.exit(0)


if __name__ == "__main__":
    main()
