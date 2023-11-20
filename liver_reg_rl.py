import gymnasium as gym
import numpy as np
import fanslicer.pycuda_simulation.segmented_volume as svol

import neurite as ne
import voxelmorph as vxm
import tensorflow as tf

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def get_all_poses(initial_pose, final_pose, num_slices, error_scale):
    coords = np.linspace(initial_pose[:-1, -1], final_pose[:-1, -1], num_slices)
    coords_with_error = coords + (np.random.randn(*np.shape(coords)) * error_scale)
    poses = []
    for coord in coords_with_error:
        initial_pose[:-1, -1] = coord
        poses.append(initial_pose)
    return poses
    
def effector_simulate_us(volume, poses):
    volume.simulate_image(poses)
    return None

def simulate_global_alignment(volume, simulated_poses, rotation_error=0.69, translation_error=64):
    for i in range(simulated_poses):
        rot = np.random.rand(3) * rotation_error
        rotation_matrix_delta = np.array([np.cos(rot[0])*np.cos(rot[2]), -np.cos(rot[2])*np.sin(rot[2]), np.sin[1]],
                                   [np.cos(rot[0])*np.sin[rot[2]]+np.sin(rot[0])*np.sin(rot[1])*np.cos(rot[2]), np.cos(rot[0])*np.cos[rot[2]]-np.sin(rot[0])*np.sin(rot[1])*np.sin(rot[2]), -np.sin(rot[0])*np.cos[1]],
                                   [np.sin(rot[0])*np.sin[rot[2]]-np.cos(rot[0])*np.sin(rot[1])*np.cos(rot[2]), np.sin(rot[0])*np.cos[rot[2]]+np.cos(rot[0])*np.sin(rot[1])*np.sin(rot[2]), np.cos(rot[0])*np.cos[1]])
        translation = np.random.rand(3) * translation_error
        simulated_poses[i][:-1, -1] += translation
        simulated_poses[i][:-1, :-1] += rotation_matrix_delta
    ct_sweep = effector_simulate_us(volume, simulated_poses)
    return ct_sweep

ct_mesh_dir = r'../ct.vtk'
us_mesh_dir = r'../us.vtk'

ct_config_dir = r'../ct_config.json'
us_config_dir = r'../us_config.json'

ct_volume = svol.SegmentedVolume(ct_mesh_dir, ct_config_dir, image_num=1, downsampling=2, voxel_size=0.5)
us_volume = svol.SegmentedVolume(us_mesh_dir, us_config_dir, image_num=1, downsampling=2, voxel_size=0.5)

ct_volumes = np.expand_dims(ct_volume, axis=0)
us_volumes = np.expand_dims(us_volume, axis=0)


# =============================================================================
# replace this part with hypernetwork training
# =============================================================================

hp_input = tf.keras.Input(shape=[2]) # one for stack length and one for regularisation
x = tf.keras.layers.Dense(32, activation='relu')(hp_input)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
hypernetwork = tf.keras.Model(hp_input, x, name='hypernetwork')

image_shape = (96, 96, 40)
model = vxm.networks.VxmDense(image_shape, int_steps=0, hyp_model=hypernetwork)

lambda_weight = hp_input
image_loss = lambda yt, yp: vxm.losses.MSE(0.05).loss(yt, yp) * (1 - lambda_weight)
gradient_loss = lambda yt, yp: vxm.losses.Grad('l2').loss(yt, yp) * lambda_weight
losses = [image_loss, gradient_loss]

###### model training #######

#############################

hypernet_model = model

# =============================================================================
# 
# =============================================================================


class RegistrationGuidance(gym.Env):
    def __init__(self, hypernet):
        
        self.ct_volumes = ct_volumes
        self.us_volumes = us_volumes
        
        self.error_scale = 5.0
        self.num_slices_us = 24
        self.num_slices_ct = 40
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(8))
        self.observation_space = gym.spaces.Box(low=0, high=1.0, shape=(96, 96, self.num_slices, 2)) # 2 because 1 ct latent and 1 simulated us
        self.rotation_error=0.69 
        self.translation_error=64
        self.step_counter = 0
        self.max_steps = 1024
        
        self.hypernet = hypernet_model
    
    def step(self, action):
        
        translation = action[:3] * 20
        rot = action[3:6] * 0.1
        regularisation = action[6]
        length = action[7] * 40
        
        rotation_matrix_delta = np.array([np.cos(rot[0])*np.cos(rot[2]), -np.cos(rot[2])*np.sin(rot[2]), np.sin[1]],
                                   [np.cos(rot[0])*np.sin[rot[2]]+np.sin(rot[0])*np.sin(rot[1])*np.cos(rot[2]), np.cos(rot[0])*np.cos[rot[2]]-np.sin(rot[0])*np.sin(rot[1])*np.sin(rot[2]), -np.sin(rot[0])*np.cos[1]],
                                   [np.sin(rot[0])*np.sin[rot[2]]-np.cos(rot[0])*np.sin(rot[1])*np.cos(rot[2]), np.sin(rot[0])*np.cos[rot[2]]+np.cos(rot[0])*np.sin(rot[1])*np.sin(rot[2]), np.cos(rot[0])*np.cos[1]])
        
        self.pose[:-1, -1] += translation
        self.pose[:-1, :-1] += rotation_matrix_delta
        
        final_translation = translation * length
        
        final_pose = np.copy(self.pose)
        final_pose[:-1, -1] = final_translation
        
        poses = get_all_poses(self.initial_pose, final_pose, self.num_slices_us, self.error_scale)
        us_sweep = effector_simulate_us(self.us_volume, poses)
        
        ct_sweep = simulate_global_alignment(self.ct_volume, poses, self.rotation_error, self.translation_error)
        
        observation = np.concatenate(np.expand_dims(us_sweep, axis=-1), np.expand_dims(ct_sweep, axis=-1), axis=-1)
        
        warped, _ = self.hypernet([us_sweep, ct_sweep, [length, regularisation]])
        
        metric = vxm.losses.Dice().loss(ct_sweep, warped) * -1
        
        reward = metric - self.current_metric
        
        self.current_metric = metric
        
        if self.step_counter >= self.max_steps:
            done = True
        else:
            done = False
        
        self.step_counter += 1
        
        return observation, reward, done, {}
        
    def reset(self):
        self.ct_volume = self.ct_volumes[np.random.randint(len(self.ct_volumes))]
        self.us_colume = self.us_volumes[np.random.randint(len(self.us_volumes))]
        
        initial_location = np.random.rand(3) * 256 
        initial_rotation = (np.random.rand(3, 3) * 2) - 1
        self.pose = np.zeros((4, 4))
        self.pose[:-1, -1] = initial_location
        self.pose[:-1, :-1] = initial_rotation
        self.pose[-1, -1] = 1
        
        action = self.action_space.sample()
        
        translation = action[:3] * 20
        rot = action[3:6] * 0.1
        regularisation = action[6]
        length = action[7] * 40
        
        rotation_matrix_delta = np.array([np.cos(rot[0])*np.cos(rot[2]), -np.cos(rot[2])*np.sin(rot[2]), np.sin[1]],
                                   [np.cos(rot[0])*np.sin[rot[2]]+np.sin(rot[0])*np.sin(rot[1])*np.cos(rot[2]), np.cos(rot[0])*np.cos[rot[2]]-np.sin(rot[0])*np.sin(rot[1])*np.sin(rot[2]), -np.sin(rot[0])*np.cos[1]],
                                   [np.sin(rot[0])*np.sin[rot[2]]-np.cos(rot[0])*np.sin(rot[1])*np.cos(rot[2]), np.sin(rot[0])*np.cos[rot[2]]+np.cos(rot[0])*np.sin(rot[1])*np.sin(rot[2]), np.cos(rot[0])*np.cos[1]])
        
        self.pose[:-1, -1] += translation
        self.pose[:-1, :-1] += rotation_matrix_delta
        
        final_translation = translation * length
        
        final_pose = np.copy(self.pose)
        final_pose[:-1, -1] = final_translation
        
        poses = get_all_poses(self.initial_pose, final_pose, self.num_slices_us, self.error_scale)
        us_sweep = effector_simulate_us(self.us_volume, poses)
        
        ct_sweep = simulate_global_alignment(self.ct_volume, poses, self.rotation_error, self.translation_error)
        
        observation = np.concatenate(np.expand_dims(us_sweep, axis=-1), np.expand_dims(ct_sweep, axis=-1), axis=-1)
        
        self.current_metric = 0
        self.step_counter = 0
        
        return observation
        

def env_maker():
    return RegistrationGuidance(hypernet_model)

vec_env = make_vec_env(env_maker, n_envs=4)
        
num_trials = 2048

model = PPO("MlpPolicy", vec_env, verbose=1) # use a custom poliy here

for trial in range(num_trials):
    print(f'Trial {trial+1} / [num_trials]:\n')
    model.learn(5120)

model.save('ppo_rl_reg_guidance')
    