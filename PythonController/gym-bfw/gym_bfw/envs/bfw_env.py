import os
import sys
import shutil
import logging
import subprocess
import random
import copy
from datetime import datetime
import numpy as np
import gym
from timeit import default_timer as timer
from gym import error, spaces, utils
from gym.utils import seeding
from . import input_reader, map_generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('BfwEnv')


class Stats:
    invalid_moves = 0
    villages_taken = 0
    villages_lost = 0
    mean_movement_range = 0

class BfwEnv(gym.Env):  
    metadata = {'render.modes': ['human']}

    game_process = None
    lua_input_file = None
    last_state = None

    state_history = {}
    last_run_stats = {0: Stats(), 1: Stats()}


    def __init__(self, **kwargs):
        self.action_space = spaces.Discrete(1+6*3) # movement range = 3
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,10,3), dtype=np.float32)

        self._parse_args(kwargs)

        self.reset_count = 0
        self.step_n = 0
        self.wesnoth_userdata_path = self._get_userdata_path()
        self.last_observation = None

        # if reset() is always called before using this environment, we dont need the game start here already
        # self._start_game()

    def _parse_args(self, kwargs):
        if 'scenario' in kwargs and kwargs.get('scenario'):
            self.scenario = kwargs.get('scenario')
        else:
            raise AttributeError('Parameter scenario is required')

        self.no_gui = True
        if 'gui' in kwargs:
            self.no_gui = not kwargs.get('gui')

        self.map_variations = 0
        if 'variations' in kwargs and kwargs.get('variations'):
            self.map_variations = kwargs.get('variations')

        self.map_rotation = 0
        if 'rotation' in kwargs and kwargs.get('rotation'):
            self.map_rotation = kwargs.get('rotation')

    def __del__(self):
        self._end_game()
 
    def step(self, action_tuple):
        player_index, action = action_tuple

        if action == -1:
            return self._get_last_observation(player_index), 0, False, {}

        return_code = self.game_process.poll()
        done = self._is_game_done(return_code)

        if player_index != 0:
            action = self._mirror_action(action)

        self._take_action(action)

        state = self._next_state()
        state.action = action

        if player_index != 0:
            state.observation = self._mirror_observation(state.observation)

        if player_index != state.side:
            logger.warn(f'Missmatch between sides of action {player_index} and observation {state.side}')
            return self.step(action_tuple)

        if state.valid:
            done = done or state.done

            self._save_game_state(state)
            reward = self._calculate_reward(state.side)
        else:
            reward = 0
            done = True

        return state.observation, reward, done, {}
 
    def reset(self):
        # Deny to many resets, specially for the case with two player ais 
        # and both want to reset the environent at the beginning
        if self.reset_count == 0 or self.step_n > 10:
            # self._end_game()

            self._calculate_stats()
            self.state_history = {}

            self._start_game()

            self.reset_count += 1
            self.step_n = 0

            return self._next_observation()
        else:
            # returning the last observation
            # a more accurate observation as return from step() with action -1
            return self.last_state.observation
 
    # openAI gym method
    # returns the current map state as string representation
    def render(self, mode='human', close=False):
        result = ''

        observation = np.transpose(self._get_last_observation(0), axes=[2,1,0])

        for terrain_row, villages_row, units_row in zip(observation[0], observation[1], observation[2]):
            result += '\n' + ('-----' * len(terrain_row)) + '-\n|'
            for terrain_cell, village_cell, unit_cell in zip(terrain_row, villages_row, units_row):
                terrain = input_reader.map_ground_index(terrain_cell)
                unit = input_reader.map_unit_index(unit_cell)
                village = input_reader.map_village_index(village_cell)

                result += f'{terrain}{village} {unit}|'

        result += '\n' + ('-----' * len(terrain_row)) + '-'

        return result

    # gets the statistics for one player
    # after a reset the statistics from the last game are available
    def get_stats(self, player_index):
        return self.last_run_stats[player_index]

    # calculate statistics from all last turns
    # helpful to have more insights into what happend in the turns
    def _calculate_stats(self):
        self.last_run_stats = {0: Stats(), 1: Stats()}

        for i in range(2):
            if not i in self.state_history: return

            player_history = self.state_history[i]

            if len(player_history) < 2: return

            invalid_moves = sum(not state.valid_move for state in player_history)
            self.last_run_stats[i].invalid_moves = invalid_moves / player_history[0].own_units

            self.last_run_stats[i].mean_movement_range = sum((state.action + 5) // 6 for state in player_history) / len(player_history)

            for n in range(len(player_history) - 1):
                diff = player_history[n + 1].own_villages - player_history[n].own_villages
                if diff > 0:
                    self.last_run_stats[i].villages_taken += diff
                elif diff < 0:
                    self.last_run_stats[i].villages_lost += diff * -1

    def _next_observation(self):
        return self._next_state().observation

    # get the last observation and adjusts it for the given player
    def _get_last_observation(self, player_index):
        result = self.last_state.observation

        # mirror the ownerships if the player index changed
        if player_index != self.last_state.side:
            result = self._mirror_observation_ownerships(result)

        # set all units to 0.5 for own or -0.5 for enemy
        result[:,:,2] *= 0.5

        # get the next moveable unit and set it to 1
        unit_x, unit_y = input_reader.get_next_unit_from_stdout(self.game_process.stdout.readline)
        result[unit_x, unit_y, 2] = 1

        # if second player mirror the observation to be up side down
        if player_index == 1:
            result = self._mirror_observation(result)
        
        # save calculations for a possible render later
        self.last_state.observation = result
        return result

    def _next_state(self):
        last_state = input_reader.read_stdout(self.game_process.stdout.readline)
        self.last_state = copy.deepcopy(last_state)
        return last_state

    def _take_action(self, action):
        lua_line = f'return {action}, {self.step_n}\n'

        self._write_line_to_file(self.lua_input_file, lua_line)

        self.step_n += 1

    def _mirror_observation(self, observation):
        return np.flip(observation, (0, 1)).copy()

    def _mirror_observation_ownerships(self, observation):
        mirror = observation.copy()
        mirror[:,:,1] *= -1
        mirror[:,:,2] *= -1
        return mirror

    def _mirror_action(self, action):
        if action == 0:
            return 0
        if ((action - 1) % 6 < 3):
            return action + 3
        else:
            return action - 3

    def _calculate_reward(self, side):
        history = self.state_history[side]
        last_state = history[-1]

        result = 0
        if not last_state.valid_move:
            result = -0.1

        if len(history) > 1:
            second_last_state = history[-2]
            gradient = last_state.own_villages - second_last_state.own_villages
            result += gradient


            first_state = history[0]
            gradient2 = (last_state.own_villages - first_state.own_villages) / len(history)
            result += gradient2

            # defender rule
            # if side == 1 and last_state.valid_move:
            #     min_own_villages = min(history, key=lambda x: x.own_villages).own_villages / last_state.all_villages
            #     result += (min_own_villages / last_state.own_units)
        
        return result

    # checks if the game process exited early
    def _is_game_done(self, return_code):
        done = False
        if return_code is not None:
            exit_message = f'Game exited with code {return_code}'
            if return_code == 0:
                logger.debug(exit_message)
            else:
                logger.error(exit_message)
            done = True
            logger.debug('Game ended')

        return done

    # starts a new game process or reuses an existing process
    # resuing is much more efficient, cause only the scenario has to be reload and not the entire process restarted
    # the game process is successfully set up, if the communication over stdout and lua file is established
    def _start_game(self):
        self._generate_new_map()

        start = timer()
        self._remove_lua_file()

        while True:
            if self.game_process is None or self.game_process.poll() is not None:
                process_args = ["wesnoth", "--multiplayer", "--exit-at-end", "--multiplayer-repeat", str(1000000), "--scenario", self.scenario]
                if self.no_gui:
                    process_args.insert(1, "--nogui")
                    process_args.insert(2, "--nosound")

                self.game_process = subprocess.Popen(process_args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

            self.lua_input_file = self._create_action_input_lua_file()

            if self.lua_input_file is None:
                logger.warn('No sync with game process possible')
                self.game_process.kill()
                self.game_process = None
            else:
                break
        
        end = timer()
        logger.info(f'Game started in {end - start}')

    # kills the game process, if still running, and deletes the lua file
    def _end_game(self):
        if self.game_process is not None:
            if self.game_process.poll() is None:
                self.game_process.kill()

        self._remove_lua_file()

    # deletes the lua file
    def _remove_lua_file(self):
        if self.lua_input_file is not None and os.path.exists(self.lua_input_file):
            os.remove(self.lua_input_file)

    # saves the game state for the given player side
    def _save_game_state(self, state):
        if state.side not in self.state_history:
            self.state_history[state.side] = []

        self.state_history[state.side].append(state)

    # generates a new map
    # if map_variations and map_rotation is set, it rotates over the given maps randomly 
    # if only map_variations is set, it generates an entire new map at the beginning and mutates it after that
    def _generate_new_map(self):
        if self.map_variations > 0:
            if self.map_rotation > 0:
                if self.reset_count % self.map_variations == 0:
                    map_index = random.randint(1,self.map_rotation)
                    self._copy_file(self.wesnoth_userdata_path + "/data/add-ons/ARL-Addon/maps/test_mini" + str(map_index) + ".map", self.wesnoth_userdata_path + "/data/add-ons/ARL-Addon/maps/test_mini.map")
                return

            if self.reset_count == 0:
                self.map = map_generator.generate_mirror(10, 10, 8)
                self._write_map_to_file()
            elif self.reset_count % self.map_variations == 0:
                self.map = map_generator.mutate_map(self.map, 1)
                self._write_map_to_file()
        
    def _write_map_to_file(self):
        map_string = map_generator.map_to_string(self.map, 10, 10)
        self._write_line_to_file(self.wesnoth_userdata_path + "/data/add-ons/ARL-Addon/maps/test_mini.map", map_string)
        date_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self._write_line_to_file("./maps/" + date_time + ".map", map_string)

    def _create_action_input_lua_file(self):
        relative_file_path = input_reader.get_input_file_from_stdout(self.game_process.stdout.readline)
        if (relative_file_path == None): return None

        lua_input_file = self.wesnoth_userdata_path + "/data" + relative_file_path.replace("~", "")
        self._write_line_to_file(lua_input_file, "return -1, -1")
        return lua_input_file

    def _write_line_to_file(self, file_name, line):
        f = open(file_name, "w")
        f.write(line)
        f.close()

    def _copy_file(self, src_file, dst_file):
        shutil.copy(src_file, dst_file)

    def _get_userdata_path(self):
        process_args = ["wesnoth", "--userdata-path"]
        stdout = subprocess.check_output(process_args)

        return stdout.decode(sys.stdout.encoding).strip()
