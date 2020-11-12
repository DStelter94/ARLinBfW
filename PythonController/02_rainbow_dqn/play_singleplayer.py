#!/usr/bin/env python3
import os
import gym
import gym_bfw
import ptan
import time
import argparse
import numpy as np
from tensorboardX import SummaryWriter

import torch

from lib import wrappers
from lib import dqn_model, common


from lib import common, main, utils

import collections

DEFAULT_ENV_NAME = "Bfw-v0"


def execute(args, params, device):
    utils.kill_game_processes()
    
    env = main.make_env(args, params)

    net1 = dqn_model.RainbowDQN(env.observation_space.shape, env.action_space.n)
    net1.load_state_dict(torch.load(args.model1, map_location=lambda storage, loc: storage))

    agent1 = ptan.agent.DQNAgent(lambda x: net1.qvals(x), ptan.actions.ArgmaxActionSelector(), device=torch.device("cpu"))

    result_name = "-" + "-rainbow" + "-scenario=" + args.scenario + "-units=" + str(args.units)
    writer1 = SummaryWriter(comment=result_name + "-player0")

    env.reset()
    
    total_reward1 = 0.0
    counter1 = collections.Counter()
    
    epsilon = 0.02
    frame_idx1 = 0

    with common.RewardTracker(writer1, 100, net1, "x.dat", 0, env) as reward_tracker1:
        
        while True:
            frame_idx1 += 1
            if np.random.random() < epsilon:
                action = [env.action_space.sample()]
            else:
                state, _, _, _ = env.step((0, -1))
                action, _ = agent1([state], [None])

            counter1[action[0]] += 1
            _, reward, done, _ = env.step((0, action[0]))
            
            total_reward1 += reward
            if done:
                reward_tracker1.reward(total_reward1, frame_idx1)
                total_reward1 = 0.0
                
                env.reset()

                net1.load_state_dict(torch.load(args.model1, map_location=lambda storage, loc: storage))

            if args.maxFrames > 0 and frame_idx1 > args.maxFrames:
                break


if __name__ == "__main__":
    params = common.HYPERPARAMS['bfw']
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m1", "--model1", required=True, help="Model player 1 file to load")
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--scenario", required=True, help="BfW scenario")
    parser.add_argument("--units", type=int, required=True, help="Number of units in the scenario")
    parser.add_argument("--variations", type=int, help="BfW map variation every n games")
    parser.add_argument("--rotation", type=int, help="BfW map rotation")
    parser.add_argument("--maxFrames", default=0, type=int, help="Max frames per train try")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    while True:
        execute(args, params, device)
