#!/usr/bin/env python3
import gym
import gym_bfw
import time
import argparse
import numpy as np

import torch

from lib import wrappers
from lib import dqn_model

import collections

DEFAULT_ENV_NAME = "Bfw-v0"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" + DEFAULT_ENV_NAME)
    args = parser.parse_args()

    env = wrappers.make_env(args.env, gui=True, scenario="side1_pass", variations=4)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
