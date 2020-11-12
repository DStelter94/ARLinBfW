#!/usr/bin/env python3
import torch
from datetime import datetime

from lib import common, main, utils


def execute(args, params, device):
    utils.kill_game_processes()
    env = main.make_env(args, params)

    result_name1, writer1, net1, tgt_net1, agent1, exp_source1, buffer1, optimizer1 = main.make_components(args, params, device, env, 0)
    result_name2, writer2, net2, tgt_net2, agent2, exp_source2, buffer2, optimizer2 = main.make_components(args, params, device, env, 1)


    frame = 0
    frame_idx1 = 0
    frame_idx2 = 0

    date_time = datetime.now().strftime("%b%d_%H-%M-%S")
    with common.RewardTracker(writer1, params['stop_reward_player1'], net1, date_time + result_name1 + ".dat", 0, env) as reward_tracker1, \
            common.RewardTracker(writer2, params['stop_reward_player2'], net2, date_time + result_name2 + ".dat", 1, env) as reward_tracker2:
        
        # fill histories
        main.train(params, buffer1, device, frame_idx1, exp_source1, reward_tracker1, optimizer1, net1, tgt_net1, writer1)
        main.train(params, buffer2, device, frame_idx2, exp_source2, reward_tracker2, optimizer2, net2, tgt_net2, writer2)
        
        while True:
            if frame // args.units % 2 == 0:
                frame_idx1 += 1
                if main.train(params, buffer1, device, frame_idx1, exp_source1, reward_tracker1, optimizer1, net1, tgt_net1, writer1):
                    break
            else:
                frame_idx2 += 1
                if main.train(params, buffer2, device, frame_idx2, exp_source2, reward_tracker2, optimizer2, net2, tgt_net2, writer2):
                    break

            frame += 1

            if args.maxFrames > 0 and frame_idx1 > args.maxFrames:
                break


if __name__ == "__main__":
    params = common.HYPERPARAMS['bfw']
    args = utils.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    
    while True:
        execute(args, params, device)
