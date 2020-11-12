import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--double", default=False, action="store_true", help="Enable double DQN")
    parser.add_argument("--scenario", required=True, help="BfW scenario")
    parser.add_argument("--units", type=int, required=True, help="Number of units in the scenario")
    parser.add_argument("--variations", type=int, help="BfW map variation every n games")
    parser.add_argument("--rotation", type=int, help="BfW map rotation")
    parser.add_argument("--maxFrames", default=0, type=int, help="Max frames per train try")
    return parser.parse_args()


def kill_game_processes():
    os.system("killall -9 wesnoth")