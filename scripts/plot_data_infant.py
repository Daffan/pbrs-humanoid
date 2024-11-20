from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    # Load data
    data_path = 'analysis/data/play_log_infant.csv'

    ### Humanoid PBRS Logging ###
    # [ 1]  Timestep
    # [54]  Agent observations
    # [18]  Agent actions (joint setpoints)
    # [13]  Floating base states in world frame
    # [ 6]  Contact forces for feet
    # [18]  Joint torques

    # plot Joint torques
    data = np.loadtxt(data_path, delimiter=',')
    timesteps = data[:400, 0]
    joint_torques = data[:400, -18:]

    fig, axs = plt.subplots(4, 4, figsize=(24, 10))
    for i in range(16):
        axs[i//4, i%4].plot(timesteps, joint_torques[:, i])
        axs[i//4, i%4].set_title(f'Joint Torque {i}')
    plt.show()

    # plot actions
    actions = data[:400, 55:73]
    fig, axs = plt.subplots(4, 4, figsize=(24, 10))
    for i in range(16):
        axs[i//4, i%4].plot(timesteps, actions[:, i])
        axs[i//4, i%4].set_title(f'Action {i}')
    plt.show()
