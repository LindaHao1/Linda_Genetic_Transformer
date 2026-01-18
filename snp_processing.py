import skrf as rf
import matplotlib.pyplot as plt
import sys
import numpy as np
# wrap this file as a function

def process_snp_file(sim_file_path, save_path, P1_SIG_IDX=0, P1_GND_IDX=1, P2_SIG_IDX=2, P2_GND_IDX=3):
# sim_file_path = "/disk/mcao53/virtuoso/EMX_work/RFIntro_pixelated_device_ec5ebcb9.work/pixelated_device_ec5ebcb9.s4p"

    net_4p = rf.Network(sim_file_path)

    # Verify points (from your error, we know it is 41)
    n_freqs = len(net_4p.frequency)
    print(f"Frequency Points: {n_freqs}")

    # ==============================================================================
    # FIX: Create the Short Circuit with correct dimensions
    # ==============================================================================
    # We need an S-matrix of shape (N_freqs, 1, 1) filled with -1.0
    s_short = np.full((n_freqs, 1, 1), -1.0 + 0j)

    # We need a Z0 array of shape (N_freqs, 1) filled with 50.0
    z0_short = np.full((n_freqs, 1), 50.0 + 0j)

    # Create the explicit network
    short = rf.Network(frequency=net_4p.frequency, s=s_short, z0=z0_short, name="short")

    # ==============================================================================
    # 3. Grounding Process
    # ==============================================================================

    # Step 1: Connect P1_GND to Short
    step1 = rf.connect(net_4p, P1_GND_IDX, short, 0)

    # Step 2: Connect P2_GND to Short
    # Calculate new index for the remaining ground port
    remaining_gnd_idx = P2_GND_IDX - 1 if P2_GND_IDX > P1_GND_IDX else P2_GND_IDX

    net_2p = rf.connect(step1, remaining_gnd_idx, short, 0)
    net_2p.name = "Renormalized_2Port"

    # ==============================================================================
    # 4. Plot Results
    # ==============================================================================
    plt.figure(figsize=(10, 6))
    net_2p.plot_s_db(m=0, n=0, label='S11 (Return Loss)')
    net_2p.plot_s_db(m=1, n=1, label='S22 (Return Loss)')
    # net_2p.plot_s_db(m=1, n=0, label='S21 (Insertion Loss)')
    plt.grid(True)
    plt.legend()
    plt.title("S-Parameters with Ground Ports Shorted")
    plt.savefig(f"{save_path}/return_loss.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    net_2p.plot_s_db(m=1, n=0, label='S21 (Insertion Loss)')
    net_2p.plot_s_db(m=0, n=1, label='S12 (Insertion Loss)')
    plt.grid(True)
    plt.legend()
    plt.title("S-Parameters with Ground Ports Shorted")
    plt.savefig(f"{save_path}/insertion_loss.png")
    plt.close()