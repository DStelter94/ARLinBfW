#!/usr/bin/env python3
import gym
import ptan
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from tensorboardX import SummaryWriter

from . import dqn_model, common, wrappers, experience

# priority replay
PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000

# C51
Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


def make_env(args, params):
    return wrappers.make_env(params['env_name'], gui=False, scenario=args.scenario, variations=args.variations, rotation=args.rotation)

def make_components(args, params, device, env, player_index):
    result_name = "-" + params['run_name'] + "-rainbow" + "-scenario=" + args.scenario + "-units=" + str(args.units) + "-player" + str(player_index) + ("-variations=" + str(args.variations) if args.variations else "") + ("-rotation=" + str(args.rotation) if args.rotation else "")
    writer = SummaryWriter(comment=result_name)

    net = dqn_model.RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    agent = ptan.agent.DQNAgent(lambda x: net.qvals(x), ptan.actions.ArgmaxActionSelector(), device=device)

    exp_source = experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=params['reward_steps'], player_index=player_index)
    buffer = ptan.experience.PrioritizedReplayBuffer(exp_source, params['replay_size'], PRIO_REPLAY_ALPHA)
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    return result_name, writer, net, tgt_net, agent, exp_source, buffer, optimizer

def calc_loss(batch, batch_weights, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)
    batch_size = len(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)

    # next state distribution
    # dueling arch -- actions from main net, distr from tgt_net

    # calc at once both next and cur states
    distr_v, qvals_v = net.both(torch.cat((states_v, next_states_v)))
    next_qvals_v = qvals_v[batch_size:]
    distr_v = distr_v[:batch_size]

    next_actions_v = next_qvals_v.max(1)[1]
    next_distr_v = tgt_net(next_states_v)
    next_best_distr_v = next_distr_v[range(batch_size), next_actions_v.data]
    next_best_distr_v = tgt_net.apply_softmax(next_best_distr_v)
    next_best_distr = next_best_distr_v.data.cpu().numpy()

    dones = dones.astype(np.bool)

    # project our distribution using Bellman update
    proj_distr = common.distr_projection(next_best_distr, rewards, dones, Vmin, Vmax, N_ATOMS, gamma)

    # calculate net output
    state_action_values = distr_v[range(batch_size), actions_v.data]
    state_log_sm_v = F.log_softmax(state_action_values, dim=1)
    proj_distr_v = torch.tensor(proj_distr).to(device)

    loss_v = -state_log_sm_v * proj_distr_v
    loss_v = batch_weights_v * loss_v.sum(dim=1)
    return loss_v.mean(), loss_v + 1e-5

def train(params, buffer, device, frame_idx, exp_source, reward_tracker, optimizer, net, tgt_net, writer):
    buffer.populate(1)
    beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

    new_rewards = exp_source.pop_total_rewards()
    if new_rewards:
        if reward_tracker.reward(new_rewards[0], frame_idx):
            return True

    if len(buffer) < params['replay_initial']:
        return False

    optimizer.zero_grad()
    batch, batch_indices, batch_weights = buffer.sample(params['batch_size'], beta)
    loss_v, sample_prios_v = calc_loss(batch, batch_weights, net, tgt_net.target_model,
                                        params['gamma'] ** params['reward_steps'], device=device)
    loss_v.backward()
    optimizer.step()
    buffer.update_priorities(batch_indices, sample_prios_v.data.cpu().numpy())

    if frame_idx % params['target_net_sync'] == 0:
        tgt_net.sync()

    return False
