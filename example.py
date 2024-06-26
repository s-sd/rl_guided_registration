import gymnasium as gym
import numpy as np

import neurite as ne
import voxelmorph as vxm
import tensorflow as tf

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from copy import deepcopy

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
tf.compat.v1.disable_eager_execution()


# =============================================================================
# Creating a dummy dataset and data generating functions (see liver_reg_rl.py for further details)
# =============================================================================

CT_vol = np.random.rand(96, 96, 96)
CT_vol_warped = np.random.rand(96, 96, 96) # this warped CT volume serves as a volume from which to sample US data (simulated US)

US_stack_size = [32, 32, 32]
CT_latent_stack_size = [32, 32, 32]

def effector_simulate_us(CT_vol_warped, start_location, US_stack_size, US_stack_len):
    # this dummy function takes in the simulated US volume i.e., the CT_vol_warped   
    # along with a start location for the US_sweep and returns the sampled simulated US_stack
    
    # stack len change not modelled in this dummy function
    
    x, y, z = start_location # max start location can be 96-1-32
    
    # in this dummy set-up we assume no rotation, for complete code see liver_reg_rl.py
    
    # sample a US stack from the simulated US volume
    US_stack = deepcopy(CT_vol_warped[x:x+US_stack_size[0], y:y+US_stack_size[1], z:z+US_stack_size[2]]) # add noise here to make it more robust (see liver_reg_rl.py for further details)
    
    return US_stack

def find_best_match(US_stack, CT_vol):
    # this is a dummy brutre force global alignment function which searches over the entire CT_vol
    # and returns the position in the CT_vol where the difference in intensity between the US_stack
    # and subarray in CT_vol is smallest
    # in summary it finds the best matching location of the US_stack inside the CT_vol
    # i.e., returns position (x, y, z) in CT_vol where the difference to US_stack is smallest
    # for real implementation use the global alignment as in the original paper
    
    A = US_stack
    B = CT_vol
    
    a_size = A.shape[0]
    b_size = B.shape[0]
    
    if a_size > b_size:
        raise ValueError("Array A must be smaller than or equal to array B in all dimensions.")
    
    min_diff = float('inf')
    best_position = (0, 0, 0)
    
    for x in range(b_size - a_size + 1):
        for y in range(b_size - a_size + 1):
            for z in range(b_size - a_size + 1):
                subarray_B = B[x:x+a_size, y:y+a_size, z:z+a_size]
                diff = np.sum((A - subarray_B) ** 2)
                
                if diff < min_diff:
                    min_diff = diff
                    best_position = (x, y, z)
    
    return best_position


def global_alignment(US_stack, CT_vol, CT_latent_stack_size):
    # this function finds creates a latent CT stack from the US stack i.e., 
    # sample the CT volume at the location obtained from global alignment
    
    position = find_best_match(US_stack, CT_vol)
    
    CT_latent_stack = CT_vol[position[0]:position[0]+CT_latent_stack_size[0], position[1]:position[1]+CT_latent_stack_size[1], position[2]:position[2]+CT_latent_stack_size[2]]
    
    return CT_latent_stack


# =============================================================================
# Creating a hypernet with two inputs (stack len and regularisation strength)
# =============================================================================

# one input for stack length and one for regularisation
hp_input = tf.keras.Input(shape=[2])
x = tf.keras.layers.Dense(32, activation='relu')(hp_input)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
hypernetwork = tf.keras.Model(hp_input, x, name='hypernetwork')

# create a simple registration network with hypernet controlling weights
image_shape = (32, 32, 32)
model = vxm.networks.VxmDense(image_shape, int_steps=0, hyp_model=hypernetwork)

# handle the regularisation in the loss
lambda_weight = hp_input
image_loss = lambda yt, yp: vxm.losses.MSE(0.05).loss(yt, yp) * (1 - lambda_weight) 
gradient_loss = lambda yt, yp: vxm.losses.Grad('l2').loss(yt, yp) * lambda_weight
losses = [image_loss, gradient_loss]

hypernet_model = model


# =============================================================================
# Creating a data generator to train the hypernet
# =============================================================================

def data_generator(CT_vol, warped_CT_vol, US_stack_size, CT_latent_stack_size):
    
    zeros = np.zeros([1, *US_stack_size, 1], dtype='float32')
    
    while True:
        # get a random start location
        US_location = [np.random.randint(warped_CT_vol.shape[0]-1-US_stack_size[0]), np.random.randint(warped_CT_vol.shape[1]-1-US_stack_size[1]), np.random.randint(warped_CT_vol.shape[2]-1-US_stack_size[2])]
        
        # get a random stack length
        US_stack_len = np.random.rand(1, 1)
        
        US_stack = effector_simulate_us(CT_vol_warped, US_location, US_stack_size, US_stack_len)
        
        CT_latent_stack = global_alignment(US_stack, CT_vol, CT_latent_stack_size)
        
        moving_image = US_stack[:, :, :, np.newaxis]
        fixed_image = CT_latent_stack[:, :, :, np.newaxis]
        
        # get a random regularisation
        regularistion = np.random.rand(1, 1)
        
        inputs = [moving_image[np.newaxis], fixed_image[np.newaxis], np.concatenate([US_stack_len, regularistion], axis=-1)]
        outputs = [fixed_image[np.newaxis], zeros[np.newaxis]]
        
        yield (inputs, outputs)
        
hypernet_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(lr=1e-4), loss=losses)
gen = data_generator(CT_vol, CT_vol_warped, US_stack_size, CT_latent_stack_size)

# train the model
history = model.fit_generator(gen, epochs=1, steps_per_epoch=8)

# =============================================================================
# Train the RL model
# =============================================================================

# we first create a dummy set of volumes for training
us_volumes = np.expand_dims(CT_vol_warped, axis=0)
ct_volumes = np.expand_dims(CT_vol, axis=0)

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
        
        self.US_stack_size = [32, 32, 32]
        self.CT_latent_stack_size = [32, 32, 32]
        
        self.start_location = None
        
        self.hypernet = hypernet_model
    
    def step(self, action):
        
        # The commented-out part below models the rotation and stack length, however,  
        # for this dummy demo, we proceed as below and only look at translation
        
        # translation = action[:3] * 20
        # rot = action[3:6] * 0.1
        # regularisation = action[6]
        # length = action[7] * 40
        
        # rotation_matrix_delta = np.array([np.cos(rot[0])*np.cos(rot[2]), -np.cos(rot[2])*np.sin(rot[2]), np.sin[1]],
        #                            [np.cos(rot[0])*np.sin[rot[2]]+np.sin(rot[0])*np.sin(rot[1])*np.cos(rot[2]), np.cos(rot[0])*np.cos[rot[2]]-np.sin(rot[0])*np.sin(rot[1])*np.sin(rot[2]), -np.sin(rot[0])*np.cos[1]],
        #                            [np.sin(rot[0])*np.sin[rot[2]]-np.cos(rot[0])*np.sin(rot[1])*np.cos(rot[2]), np.sin(rot[0])*np.cos[rot[2]]+np.cos(rot[0])*np.sin(rot[1])*np.sin(rot[2]), np.cos(rot[0])*np.cos[1]])
        
        # self.pose[:-1, -1] += translation
        # self.pose[:-1, :-1] += rotation_matrix_delta
        
        # final_translation = translation * length
        
        # final_pose = np.copy(self.pose)
        # final_pose[:-1, -1] = final_translation
        
        # poses = get_all_poses(self.initial_pose, final_pose, self.num_slices_us, self.error_scale)
        
        # the rl controller outputs new stack location guidance
        translation = action[:3] * 10
        self.start_location += translation
        
        # the rl controller also outputs suggestions for regularisation and stack length
        regularisation = action[6]
        length = action[7] * 40
        
        # the effectors uses this guidance to get a new us stack
        us_stack = effector_simulate_us(self.us_volume, self.start_location, self.US_stack_size, US_stack_len=None)
        
        # the global alignment gets the corresponding ct latent stack
        ct_latent_stack = global_alignment(us_stack, self.ct_vol, self.CT_latent_stack_size)
        
        observation = np.concatenate(np.expand_dims(us_stack, axis=-1), np.expand_dims(ct_latent_stack, axis=-1), axis=-1)
        
        # the hyperent warps the us_stack to be aligned with the ct latent stack
        warped, _ = self.hypernet([us_stack, ct_latent_stack, np.concatenate([length, regularisation], axis=-1)])
        
        # we now compute performance of the network by comparing warped us stack with fixed ct latent stack
        metric = vxm.losses.Dice().loss(ct_latent_stack, warped) * -1
        
        # the reward is the performance improvement compared to previous iteration
        reward = metric - self.current_metric
        
        self.current_metric = metric
        
        if self.step_counter >= self.max_steps:
            done = True
        else:
            done = False
        
        self.step_counter += 1
        
        return observation, reward, done, {}
        
    def reset(self):
        
        # at each episode we sample a new pre (ct) and simulated intra-operative (us) volume
        index = np.random.randint(len(self.ct_volumes))
        self.ct_volume = self.ct_volumes[index]
        self.us_colume = self.us_volumes[index]
        
        # we initialise at a random start location in the us volume
        self.start_location = [np.random.randint(self.us_volume.shape[0]-1-US_stack_size[0]), np.random.randint(self.us_volume.shape[1]-1-US_stack_size[1]), np.random.randint(self.us_volume.shape[2]-1-US_stack_size[2])]
        
        # the effector gets the us stack at the location
        us_stack = effector_simulate_us(self.us_volume, self.start_location, self.US_stack_size, US_stack_len=None)
        
        # the ct latent stack is sampled from the global alignment function
        ct_latent_stack = global_alignment(us_stack, self.ct_vol, self.CT_latent_stack_size)
        
        # and observation is generated for the rl controller
        observation = np.concatenate(np.expand_dims(us_stack, axis=-1), np.expand_dims(ct_latent_stack, axis=-1), axis=-1)
        
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
