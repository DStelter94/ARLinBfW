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

def play_round(net, state, counter, total_reward):
    state_v = torch.tensor(np.array([state], copy=False))
    q_vals = net(state_v).data.numpy()[0]
    action = np.argmax(q_vals)
    counter[action] += 1
    state, reward, done, _ = env.step(action)
    total_reward += reward
    if done:
        return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m1", "--model1", required=True, help="Model player 1 file to load")
    parser.add_argument("-m2", "--model2", required=True, help="Model player 2 file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" + DEFAULT_ENV_NAME)
    args = parser.parse_args()

    env = wrappers.make_env(args.env, gui=True, scenario="multi_side_ai")
    net1 = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    net2 = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    net1.load_state_dict(torch.load(args.model1, map_location=lambda storage, loc: storage))
    net2.load_state_dict(torch.load(args.model2, map_location=lambda storage, loc: storage))

    state1 = env.reset()
    state2 = state1
    total_reward1 = 0.0
    total_reward2 = 0.0
    counter1 = collections.Counter()
    counter2 = collections.Counter()
    
    epsilon = 0.2
    frame = 0
    while True:
        if frame % 2 == 0:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_v = torch.tensor(np.array([state1], copy=False))
                q_vals = net1(state_v).data.numpy()[0]
                action = np.argmax(q_vals)

            counter1[action] += 1
            state1, reward, done, _ = env.step((0, action))
            total_reward1 += reward
            if done:
                break


            # if play_round(net1, state1, c1, total_reward1):
            #     break
        else:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_v = torch.tensor(np.array([state2], copy=False))
                q_vals = net2(state_v).data.numpy()[0]
                action = np.argmax(q_vals)

            counter2[action] += 1
            state2, reward, done, _ = env.step((1, action))
            total_reward2 += reward
            if done:
                break

            # if play_round(net2, state2, c2, total_reward2):
            #     break
        frame += 1

    print("Total reward player1: %.2f" % total_reward1)
    print("Total reward player2: %.2f" % total_reward2)
    print("Action counts player1:", counter1)
    print("Action counts player2:", counter2)
