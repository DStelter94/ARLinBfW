import sys
import time
import numpy as np
import torch
import torch.nn as nn


HYPERPARAMS = {
    'bfw': {
        'env_name':         "Bfw-v0",
        'stop_reward_player2':  100.0,
        'stop_reward_player1':  100.0,
        'run_name':         'bfw',
        'replay_size':      100000,
        'replay_initial':   10000,
        'target_net_sync':  10000,
        'epsilon_frames':   100000,
        'epsilon_start':    1.0,
        'epsilon_final':    0.05,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32,
        'reward_steps':     2,
    }
}


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def calc_loss_dqn(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


class RewardTracker:
    def __init__(self, writer, stop_reward, net, file_name, player_index, env):
        self.writer = writer
        self.stop_reward = stop_reward
        self.net = net
        self.file_name = file_name
        self.best_mean_reward = None
        self.player_index = player_index
        self.env = env

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        self.total_invalid_moves = []
        self.total_mean_movement_range = []
        self.total_villages_taken = []
        self.total_villages_lost = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, player %d, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), self.player_index + 1, mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        
        self.write_stats(frame)

        if self.best_mean_reward is None or self.best_mean_reward < mean_reward:
            torch.save(self.net.state_dict(), "./nets/" + self.file_name)
            self.best_mean_reward = mean_reward

        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False

    def write_stats(self, frame):
        player_stats = self.env.get_stats(self.player_index)
        
        self.total_invalid_moves.append(player_stats.invalid_moves)
        self.total_mean_movement_range.append(player_stats.mean_movement_range)
        self.total_villages_taken.append(player_stats.villages_taken)
        self.total_villages_lost.append(player_stats.villages_lost)

        self.writer.add_scalar("invalid_moves_per_unit", player_stats.invalid_moves, frame)
        self.writer.add_scalar("villages_taken", player_stats.villages_taken, frame)
        self.writer.add_scalar("villages_lost", player_stats.villages_lost, frame)
        self.writer.add_scalar("mean_movement_range", player_stats.mean_movement_range, frame)
        
        self.writer.add_scalar("invalid_moves_per_unit_100", np.mean(self.total_invalid_moves[-100:]), frame)
        self.writer.add_scalar("villages_taken_100", np.mean(self.total_villages_taken[-100:]), frame)
        self.writer.add_scalar("villages_lost_100", np.mean(self.total_villages_lost[-100:]), frame)
        self.writer.add_scalar("mean_movement_range_100", np.mean(self.total_mean_movement_range[-100:]), frame)
        
        return


class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self.epsilon_frames = params['epsilon_frames']
        self.frame(0)

    def frame(self, frame):
        self.epsilon_greedy_selector.epsilon = \
            max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)


def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    """
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)
    for atom in range(n_atoms):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * delta_z) * gamma))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
    if dones.any():
        proj_distr[dones] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones]))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones.copy()
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones.copy()
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return proj_distr
