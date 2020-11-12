#!/usr/bin/env python3
import gym
import ptan
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tensorboardX import SummaryWriter

from . import dqn_model, common, wrappers, experience

STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 100


def make_env(args, params):
    return wrappers.make_env(params['env_name'], gui=False, scenario=args.scenario, variations=args.variations, rotation=args.rotation)


def make_components(args, params, device, env, player_index):
    result_name = "-" + params['run_name'] + "-%d-step" % params['reward_steps'] + "-double=" + str(args.double) + "-scenario=" + args.scenario + "-units=" + str(args.units) + "-player" + str(player_index) + ("-variations=" + str(args.variations) if args.variations else "") + ("-rotation=" + str(args.rotation) if args.rotation else "")
    writer = SummaryWriter(comment=result_name)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=params['reward_steps'], player_index=player_index)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    return result_name, writer, net, tgt_net, selector, epsilon_tracker, agent, exp_source, buffer, optimizer


def calc_loss(batch, net, tgt_net, gamma, device="cpu", double=True):
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    if double:
        next_state_actions = net(next_states_v).max(1)[1]
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    else:
        next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def calc_values_of_states(states, net, device="cpu"):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)


def train(args, params, device, buffer, epsilon_tracker, frame_idx, exp_source, reward_tracker, selector, optimizer, net, tgt_net, writer, eval_states):
    buffer.populate(1)
    epsilon_tracker.frame(frame_idx)

    new_rewards = exp_source.pop_total_rewards()
    if new_rewards:
        if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
            return True

    if len(buffer) < params['replay_initial']:
        return False
    if eval_states is None:
        eval_states = buffer.sample(STATES_TO_EVALUATE)
        eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
        eval_states = np.array(eval_states, copy=False)

    optimizer.zero_grad()
    batch = buffer.sample(params['batch_size'])
    loss_v = calc_loss(batch, net, tgt_net.target_model, gamma=params['gamma']**params['reward_steps'], device=device,
                        double=args.double)
    loss_v.backward()
    # https://stackoverflow.com/questions/47036246/dqn-q-loss-not-converging
    # for param in net.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()

    writer.add_scalar("mean_mse_loss", torch.mean(loss_v).item(), frame_idx)

    if frame_idx % params['target_net_sync'] == 0:
        tgt_net.sync()
    if frame_idx % EVAL_EVERY_FRAME == 0:
        mean_val = calc_values_of_states(eval_states, net, device=device)
        writer.add_scalar("values_mean", mean_val, frame_idx)

    return False
