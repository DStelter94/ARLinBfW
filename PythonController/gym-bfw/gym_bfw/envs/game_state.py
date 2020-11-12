import numpy as np

class GameState:
    def __init__(self):
        self.valid = False

        self.observation = np.zeros((10, 10, 3), np.float32)
        self.valid_move = False
        self.all_villages = 0
        self.own_villages = 0
        self.own_units = 0
        self.turn = 0
        self.turn_max = 0
        self.gold = 0
        self.side = -1
        self.done = False
        self.action = 0