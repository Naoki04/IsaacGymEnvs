# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
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

import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from .base.vec_task import VecTask

class Cartpole(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.reset_dist = self.cfg["env"]["resetDist"]

        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.max_episode_length = 500

        self.cfg["env"]["numObservations"] = 4
        self.cfg["env"]["numActions"] = 1

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self._actors_root_state = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        
        
        

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        
        prim_names = ["robot", "sphere"]
        self.gym_assets = dict.fromkeys(prim_names)
        self.gym_indices = dict.fromkeys(prim_names)
        self.gym_indices["robot"] = []
        self.gym_indices["sphere"] = []
        
        
        
        # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0) if self.up_axis == 'z' else gymapi.Vec3(0.5 * -spacing, 0.0, -spacing)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/cartpole.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True # True
        asset_options.flip_visual_attachments = True # True
        #cartpole_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.gym_assets["robot"] = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(self.gym_assets["robot"])

        pose = gymapi.Transform()
        if self.up_axis == 'z':
            pose.p.z = 2.0
            # asset is rotated z-up by default, no additional rotations needed
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        else:
            pose.p.y = 2.0
            pose.r = gymapi.Quat(-np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)
            
        # sphereのassetを作成
        self.ball_radius = 0.1
        ball_options = gymapi.AssetOptions()
        ball_options.density = 200
        self.gym_assets["sphere"] = self.gym.create_sphere(self.sim, self.ball_radius, ball_options)

        self.cartpole_handles = []
        self.sphere_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            cartpole_handle = self.gym.create_actor(env_ptr, self.gym_assets["robot"], pose, "cartpole", i, 1, 0)

            dof_props = self.gym.get_actor_dof_properties(env_ptr, cartpole_handle)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, cartpole_handle, dof_props)            
            
            
            ball_pose = gymapi.Transform()
            ball_pose.p.x = 0.2
            ball_pose.p.z = 2.0
            ball_handle = self.gym.create_actor(env_ptr, self.gym_assets["sphere"], ball_pose, "ball", i, 0, 0)
            
            cartpole_idx = self.gym.get_actor_index(env_ptr, cartpole_handle, gymapi.DOMAIN_SIM)
            self.gym_indices["robot"].append(cartpole_idx)
            sphere_idx = self.gym.get_actor_index(env_ptr, ball_handle, gymapi.DOMAIN_SIM)
            self.gym_indices["sphere"].append(sphere_idx)
            
            self.envs.append(env_ptr)
            
            # convert gym indices from list to tensor
        for asset_name, asset_indices in self.gym_indices.items():
            self.gym_indices[asset_name] = torch.tensor(asset_indices, dtype=torch.long, device=self.device)
            
        
        

    def compute_reward(self):
        # retrieve environment observations from buffer
        pole_angle = self.obs_buf[:, 2]
        pole_vel = self.obs_buf[:, 3]
        cart_vel = self.obs_buf[:, 1]
        cart_pos = self.obs_buf[:, 0]

        self.rew_buf[:], self.reset_buf[:] = compute_cartpole_reward(
            pole_angle, pole_vel, cart_vel, cart_pos,
            self.reset_dist, self.reset_buf, self.progress_buf, self.max_episode_length
        )

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)

        self.obs_buf[env_ids, 0] = self.dof_pos[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 1] = self.dof_vel[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 2] = self.dof_pos[env_ids, 1].squeeze()
        self.obs_buf[env_ids, 3] = self.dof_vel[env_ids, 1].squeeze()

        return self.obs_buf

    def reset_idx(self, env_ids):
        
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        
        # ロボットのDOFの初期状態を設定
        positions = 0.2 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
        velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
        
        # ロボットのroot_stateの(初期座標・回転・速度・角速度)を設定
        robot_positions = 0.2 * (torch.rand((len(env_ids), 3), device=self.device) - 0.5)
        robot_positions[:, 2] = 1.0
        robot_rotations = torch.zeros((len(env_ids), 4), device=self.device)
        robot_rotations[:, 3] = 1.0
        robot_velocities = torch.zeros((len(env_ids), 3), device=self.device)
        robot_angular_velocities = torch.zeros((len(env_ids), 3), device=self.device)
        
        robot_root_state = torch.cat([robot_positions, robot_rotations, robot_velocities, robot_angular_velocities], dim=1) # (envs, 13)
        # 球のroot_state(初期座標・回転・速度・角速度)を設定
        sphere_positions = 0.2 * (torch.rand((len(env_ids), 3), device=self.device) - 0.5)*1.5
        sphere_positions[:, 2] = 1.0
        sphere_rotations = torch.zeros((len(env_ids), 4), device=self.device)
        sphere_rotations[:, 3] = 1.0
        sphere_velocities = torch.zeros((len(env_ids), 3), device=self.device)
        sphere_angular_velocities = torch.zeros((len(env_ids), 3), device=self.device)
        
        sphere_root_state = torch.cat([sphere_positions, sphere_rotations, sphere_velocities, sphere_angular_velocities], dim=1) # (envs, 13)
        # すべての初期状態を結合して、_actors_root_stateに設定 (envs*2, 13)
        self._actors_root_state[env_ids, :] = robot_root_state
        self._actors_root_state[len(env_ids)+env_ids, :] = sphere_root_state
       
        

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]
        
        
        # インデックスを作成(前半がロボット、後半が球)
        robot_indices = self.gym_indices["robot"][env_ids].to(torch.int32)
        sphere_indices = self.gym_indices["sphere"][env_ids].to(torch.int32)
        all_indices = torch.unique(torch.cat([robot_indices, sphere_indices]))
        
        # ロボットのDOFの初期状態を反映
        self.gym.set_dof_state_tensor_indexed(
                                            self.sim, 
                                            gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(robot_indices), 
                                            len(robot_indices)
                                            )
        # ロボット・球のroot_stateの初期状態を反映
        self.gym.set_actor_root_state_tensor_indexed(
                                            self.sim,
                                            gymtorch.unwrap_tensor(self._actors_root_state),
                                            gymtorch.unwrap_tensor(all_indices),
                                            len(all_indices)
                                        )
                                        
        
        
        
    def pre_physics_step(self, actions):
        actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        actions_tensor[::self.num_dof] = actions.to(self.device).squeeze() * self.max_push_effort
        forces = gymtorch.unwrap_tensor(actions_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_cartpole_reward(pole_angle, pole_vel, cart_vel, cart_pos,
                            reset_dist, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # reward is combo of angle deviated from upright, velocity of cart, and velocity of pole moving
    reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)

    # adjust reward for reset agents
    reward = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reward) * -2.0, reward)
    reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

    #reset = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reset_buf), reset_buf)
    #reset = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reset_buf), reset)
    #reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    # とりあえず、最大エピソード長に達したらリセットするようにする
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return reward, reset
