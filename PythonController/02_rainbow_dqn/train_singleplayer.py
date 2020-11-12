#!/usr/bin/env python3
import torch
from datetime import datetime

from lib import common, main, utils


def execute(args, params, device):
    utils.kill_game_processes()
    env = main.make_env(args, params)

    result_name, writer, net, tgt_net, agent, exp_source, buffer, optimizer = main.make_components(args, params, device, env, 0)

    frame_idx = 0

    date_time = datetime.now().strftime("%b%d_%H-%M-%S")
    with common.RewardTracker(writer, params['stop_reward_player1'], net, date_time + result_name + ".dat", 0, env) as reward_tracker:
        while True:
            frame_idx += 1
            if main.train(params, buffer, device, frame_idx, exp_source, reward_tracker, optimizer, net, tgt_net, writer):
                break

            if args.maxFrames > 0 and frame_idx > args.maxFrames:
                break


if __name__ == "__main__":
    params = common.HYPERPARAMS['bfw']
    args = utils.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    while True:
        execute(args, params, device)
