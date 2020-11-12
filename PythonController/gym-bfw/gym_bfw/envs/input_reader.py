import re
import numpy as np
import logging
from . import game_state as gs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_input_file_from_stdout(read_func):
    for line in iter(read_func, b''):
        line = line.decode('utf-8').rstrip()

        if (line.startswith('**inputPath*')):
            m = re.match(r'\*{2}inputPath\*(.*)', line)
            input_path = m.group(1)
            return input_path

def get_next_unit_from_stdout(read_func):
    for line in iter(read_func, b''):
        line = line.decode('utf-8').rstrip()
        
        if (line.startswith('**log*')):
            m = re.match(r'\*{2}log\*(.*)', line)
            lua_log = m.group(1)
            logger.info('Lua logging %s', lua_log)
        if (line.startswith('**next_unit*')):
            m = re.match(r'\*{2}next_unit\*(.*),(.*)', line)
            unit_x = int(m.group(1))
            unit_y = int(m.group(2))
            return unit_x - 1, unit_y - 1

def read_stdout(read_func):
    start = False

    state = gs.GameState()
    lines = ""

    for line in iter(read_func, b''):
        line = line.decode('utf-8').rstrip()
        lines = lines + "\n" + line 

        if ('***start***' == line):
            start = True
            continue

        elif (line.startswith('**log*')):
            m = re.match(r'\*{2}log\*(.*)', line)
            lua_log = m.group(1)
            logger.info('Lua logging %s', lua_log)

        if (start):
            if ('***end***' ==  line):
                start = False
                state.valid = True
                break

            elif (line.startswith('**map*')):
                m = re.match(r'\*{2}map\*(.*),(.*),(.*),(.*),(.*)', line)
                x_pos = m.group(1)
                y_pos = m.group(2)
                ground = m.group(3)
                owner = m.group(4)
                unit = m.group(5)

                ground_index = map_ground(ground)
                unit_index = int(unit)
                owner_index = int(owner)

                state.observation[int(x_pos)-1, int(y_pos)-1] = [ground_index, owner_index, unit_index]

            elif (line.startswith('**valid_move*')):
                m = re.match(r'\*{2}valid_move\*(.*)', line)
                state.valid_move = m.group(1) == 'true'

            elif (line.startswith('**finished*')):
                m = re.match(r'\*{2}finished\*(.*)', line)
                state.done = m.group(1) == 'true'

            elif (line.startswith('**turn*')):
                m = re.match(r'\*{2}turn\*(.*)', line)
                state.turn = int(m.group(1))
                
            elif (line.startswith('**turn_max*')):
                m = re.match(r'\*{2}turn_max\*(.*)', line)
                state.turn_max = int(m.group(1))

            elif (line.startswith('**gold*')):
                m = re.match(r'\*{2}gold\*(.*)', line)
                state.gold = int(m.group(1))

            elif (line.startswith('**all_villages*')):
                m = re.match(r'\*{2}all_villages\*(.*)', line)
                state.all_villages = int(m.group(1))

            elif (line.startswith('**own_villages*')):
                m = re.match(r'\*{2}own_villages\*(.*)', line)
                state.own_villages = int(m.group(1))

            elif (line.startswith('**own_units*')):
                m = re.match(r'\*{2}own_units\*(.*)', line)
                state.own_units = int(m.group(1))            

            elif (line.startswith('**side*')):
                m = re.match(r'\*{2}side\*(.*)', line)
                state.side = int(m.group(1)) - 1

    return state

def map_ground(ground_str):
    if ground_str == 'Gg^Ve':
        return 1
    elif ground_str == 'Gg':
        return 0.5
    elif ground_str == 'Ch':
        return 0

def map_ground_index(ground_idx):
    if ground_idx == 1:
        return 'V'
    elif ground_idx == 0.5:
        return ' '
    elif ground_idx == 0:
        return 'C'

def map_unit_index(unit_idx):
    if unit_idx == 1:
        return 'X'
    if unit_idx == 0.5:
        return 'O'
    elif unit_idx == -0.5:
        return 'E'
    elif unit_idx == 0:
        return ' '

def map_village_index(village_idx):
    if village_idx == 1:
        return 'o'
    elif village_idx == -1:
        return 'e'
    elif village_idx == 0:
        return ' '
