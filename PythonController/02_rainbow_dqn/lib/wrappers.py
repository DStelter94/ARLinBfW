import cv2
import gym
import gym.spaces
import gym_bfw
import numpy as np
import collections


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

def make_env(env_name, **kwargs):
    env = gym.make(env_name, **kwargs)
    env = ImageToPyTorch(env)
    return env
