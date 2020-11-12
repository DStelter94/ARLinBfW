#!/usr/bin/env python3
import ptan
import torch
from datetime import datetime

from lib import common, main, utils

NET_SYNC = 10000


def execute(args, params, device):
    utils.kill_game_processes()
    env = main.make_env(args, params)

    result_name1, writer1, net1, tgt_net1, selector1, epsilon_tracker1, agent1, exp_source1, buffer1, optimizer1 = main.make_components(args, params, device, env, 0)
    net2 = ptan.agent.TargetNet(net1)
    agent2 = ptan.agent.DQNAgent(net2.target_model, ptan.actions.ArgmaxActionSelector(), device=device)

    frame = 0
    frame_idx1 = 0
    eval_states1 = None

    date_time = datetime.now().strftime("%b%d_%H-%M-%S")
    with common.RewardTracker(writer1, params['stop_reward_player1'], net1, date_time + result_name1 + ".dat", 0, env) as reward_tracker1:

        # fill history
        main.train(args, params, device, buffer1, epsilon_tracker1, frame_idx1, exp_source1, reward_tracker1, selector1, optimizer1, net1, tgt_net1, writer1, eval_states1)

        while True:
            if frame // args.units % 2 == 0:
                state, _, _, _  = env.step((1, -1))
                action, _ = agent2([state])
                state, reward, done, _ = env.step((1, action[0]))
                if done:
                    state = env.reset()
            else:
                frame_idx1 += 1
                if main.train(args, params, device, buffer1, epsilon_tracker1, frame_idx1, exp_source1, reward_tracker1, selector1, optimizer1, net1, tgt_net1, writer1, eval_states1):
                    break

            if args.maxFrames > 0 and frame_idx1 > args.maxFrames:
                break

            frame += 1
            if frame % NET_SYNC == 0:
                net2.sync()


if __name__ == "__main__":
    params = common.HYPERPARAMS['bfw']
    args = utils.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    
    while True:
        execute(args, params, device)
    