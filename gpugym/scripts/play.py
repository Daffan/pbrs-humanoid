# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from gpugym import LEGGED_GYM_ROOT_DIR
import os
import imageio
import json
import time

import isaacgym
from gpugym.envs import *
from gpugym.utils import  get_args, export_policy, export_critic, task_registry, Logger, get_load_path

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1  # min(env_cfg.env.num_envs, 16)
    env_cfg.env.render_all_envs = True
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = True
    env_cfg.control.torque_scale = 1.0
    env_cfg.commands.resampling_time = -1
    env_cfg.commands.ranges.lin_vel_x = [0.2, 0.6]
    env_cfg.commands.ranges.lin_vel_y = [0.2, 0.6]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False #True
    env_cfg.domain_rand.push_interval_s = 2
    env_cfg.domain_rand.max_push_vel_xy = 1.0
    env_cfg.init_state.reset_ratio = 0.8
    train_cfg.runner.load_run = args.load_run if args.load_run is not None else -1

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy(ppo_runner.alg.actor_critic, path)
        print('Exported policy model to: ', path)

    # export critic as a jit module (used to run it from C++)
    if EXPORT_CRITIC:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'critics')
        export_critic(ppo_runner.alg.actor_critic, path)
        print('Exported critic model to: ', path)

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 2  # which joint is used for logging
    stop_state_log = env.max_episode_length-1  # number of steps before plotting states
    stop_rew_log = env.max_episode_length-1  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    video_frames = []

    command_resample_time = 100
    # schedule 1
    # commands = [
    #     [0.1, 0., 0., 0.],
    #     [0.3, 0., 0., 0.],
    #     [0.5, 0., 0., 0.],
    #     [0.7, 0., 0., 0.],
    #     [0.9, 0., 0., 0.],
    #     [1.1, 0., 0., 0.],
    # ]

    # schedule 2
    commands = [
        [0.5, 0., 0., 0.],
    ]


    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    load_path = get_load_path(log_root, train_cfg.runner.load_run, -1)
    load_path_dir = os.path.dirname(load_path)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    load_path_dir = os.path.join(load_path_dir, timestamp)
    os.makedirs(f'{load_path_dir}/videos', exist_ok=True)
    os.makedirs(f'{load_path_dir}/data', exist_ok=True)
    print(f"Saving videos to {load_path_dir}/videos")

    play_log = []
    for i in range(int(env.max_episode_length)+1):
        if i % command_resample_time == 0:
            # env_ids = torch.arange(env_cfg.env.num_envs)
            # env._resample_commands(env_ids)
            if len(commands) > 0:
                tensor_commands = torch.tensor(commands.pop(), dtype=torch.float32).to(env.device)
                env.commands = tensor_commands.unsqueeze(0).repeat(env_cfg.env.num_envs, 1)
                print(f"Resampled commands at step {i}")
                print(f"Commands: {env.commands}")

        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        video_frames.append(env.render_all_envs())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            ### Humanoid PBRS Logging ###
            # [ 1]  Timestep
            # [38]  Agent observations
            # [10]  Agent actions (joint setpoints)
            # [13]  Floating base states in world frame
            # [ 6]  Contact forces for feet
            # [10]  Joint torques
            # [3]   Left Foot positions
            # [3]   Right Foot positions
            play_log.append(
                [i*env.dt]
                + obs[robot_index, :].cpu().numpy().tolist()
                + actions[robot_index, :].detach().cpu().numpy().tolist()
                + env.root_states[0, :].detach().cpu().numpy().tolist()
                + env.contact_forces[robot_index, env.end_eff_ids[0], :].detach().cpu().numpy().tolist()
                + env.contact_forces[robot_index, env.end_eff_ids[1], :].detach().cpu().numpy().tolist()
                + env.torques[robot_index, :].detach().cpu().numpy().tolist()
                + env.foot_pos[robot_index, 0].detach().cpu().numpy().tolist()
                + env.foot_pos[robot_index, 1].detach().cpu().numpy().tolist()
            )
        elif i==stop_state_log:
            print("Saving play log to ", f'{load_path_dir}/data/play_log.csv')
            np.savetxt(f'{load_path_dir}/data/play_log.csv', play_log, delimiter=',')
            # logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                # if num_episodes>0:
                    # logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

    video_frames = np.stack(video_frames).astype(np.uint8)
    video_frames = video_frames.transpose(1, 0, 2, 3, 4)
    fps = 25

    for i, frames in enumerate(video_frames):
        print(f'Processing env {i} of {len(video_frames)}', end="\r")
        output_video_path = os.path.join(f"{load_path_dir}/videos", f'env_{i}.mp4')
        video_writer = imageio.get_writer(output_video_path, fps=fps)
        for frame in frames:
            video_writer.append_data(frame)
        video_writer.close()

if __name__ == '__main__':
    EXPORT_POLICY = True
    EXPORT_CRITIC = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
