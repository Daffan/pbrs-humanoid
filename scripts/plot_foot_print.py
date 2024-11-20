from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

if __name__ == '__main__':
    # Load data
    data_path = 'logs/PBRS_HumanoidLocomotion/Sep22_22-43-47_pbrs_humanoid/20240923-082004/data/play_log.csv'
    ### Humanoid PBRS Logging ###
    # [ 1]  Timestep
    # [38]  Agent observations
    # [10]  Agent actions (joint setpoints)
    # [13]  Floating base states in world frame
    # [ 6]  Contact forces for feet
    # [10]  Joint torques
    # [3]   Left Foot positions
    # [3]   Right Foot positions

    # plot Joint torques
    data = np.loadtxt(data_path, delimiter=',')
    timesteps = data[:, 0]
    foot_pos_left = data[:, -6:-3]
    foot_pos_right = data[:, -3:]
    contact_forces_left = data[:, 62:65]
    contact_forces_right = data[:, 65:68]
    commands = data[:, 11:14]

    # scatter plot in one figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # only plot when in contact
    in_contact_left = np.sum(np.abs(contact_forces_left), axis=1) > 0
    in_contact_right = np.sum(np.abs(contact_forces_right), axis=1) > 0

    cmap = sns.cubehelix_palette(as_cmap=True)
    unique_commands = np.unique(commands[:, 0])
    print(unique_commands)
    
    idx_left = in_contact_left
    idx_right = in_contact_right
    # use uc[0] vx command as the color map

    a1 = ax.scatter(foot_pos_left[idx_left, 0], foot_pos_left[idx_left, 1], c=commands[idx_left, 0], cmap=cmap)
    a2 = ax.scatter(foot_pos_right[idx_right, 0], foot_pos_right[idx_right, 1], c=commands[idx_right, 0], cmap=cmap)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    # put colorbar at the bottom with title "vx command"
    cbar = fig.colorbar(a1, ax=ax, orientation='horizontal')
    cbar.set_label('vx')
    plt.show()