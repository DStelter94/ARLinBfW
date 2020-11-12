### example map ###
# Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg
# Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg^Ve, Gg
# Gg, Gg, Gg, Gg^Ve, Gg, Gg, 1 Ch, Gg, Gg, Gg, Gg, Gg
# Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg
# Gg, Gg, Gg^Ve, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg
# Gg, Gg, Gg, Gg, Gg, Gg^Ve, Gg^Ve, Gg, Gg, Gg, Gg, Gg
# Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg^Ve, Gg, Gg
# Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg
# Gg, Gg^Ve, Gg, Gg, Gg, Gg, Gg, Gg, Gg^Ve, Gg, Gg, Gg
# Gg, Gg, Gg, Gg, Gg, 2 Ch, Gg, Gg, Gg, Gg, Gg, Gg
# Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg
# Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg, Gg

import random

# Old map generator
def generate(width, height, number_of_villages):
    map = ["Gg"] * (width * height)
    map[15] = "1 Ch"
    map[84] = "2 Ch"

    count = 0
    while count < number_of_villages:
        index = random.randint(0, width*height-1)
        if (index != 15 and index != 84):
            map[index] = "Gg^Ve"
            count += 1

    return map

# Generates horizontal and vertical mirrored map
def generate_mirror(width, height, number_of_villages):
    assert height % 2 == 0
    assert width % 2 == 0
    map_size = width * height

    map = ["Gg"] * map_size

    castle_index = random.randint(0, map_size / 2 - 1)
    map[castle_index] = "1 Ch"
    map[map_size - 1 - castle_index] = "2 Ch"

    count = 0
    while count < number_of_villages / 2:
        index = random.randint(0, map_size / 2 - 1)
        if (map[index] != "1 Ch"):
            map[index] = "Gg^Ve"
            map[map_size - 1 - index] = "Gg^Ve"
            count += 1
    
    return map

# mutates the given map by given number
def mutate_map(map, mutations):
    for _ in range(mutations):
        index1 = random.randint(0, len(map) / 2 - 1)
        index2 = random.randint(0, len(map) / 2 - 1)
        temp = map[index1]
        map[index1] = map[index2]
        map[index2] = temp

        map_size = len(map)
        temp = map[map_size - 1 - index1]
        map[map_size - 1 - index1] = map[map_size - 1 - index2]
        map[map_size - 1 - index2] = temp

    return map

# Converts map array to string and adds borders
def map_to_string(map, width, height):
    string = ", ".join(["Gg"] * (width+2)) + '\n' #top border
    for y in range(height):
        row = map[y*width:y*width+width]
        row.insert(0, "Gg") #left border
        row.append("Gg") #right border
        string +=  ", ".join(row) + '\n'

    string += ", ".join(["Gg"] * (width+2)) #button border

    return string