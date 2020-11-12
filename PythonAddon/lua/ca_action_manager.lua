-- for debugging this lua code per consule in the external python code use following line
--      std_print('**log*...logMessage...')

local ca_action_manager = {}

math.randomseed( os.time() + os.clock() * 1000)

-- inits the file communication between the CAs and an external script once
-- evaluates the next action
-- currently executes one action per turn and player
-- https://wiki.wesnoth.org/Creating_Custom_AIs#Evaluation_Function
function ca_action_manager:evaluation(cfg, data)    
    -- first run of this ca -> game start
    -- wml.variables are the same for all CAs and so for all players, so this part only triggers once per game
    if (wml.variables.ai_input_path == nil) then
        init_and_wait_for_input_file_creation()
        
        wml.variables.ai_round_counter = -1

        stdout_observation(true)
    end

    -- already did one move for all units this turn
    if all_units_moved() then
        return 0
    end

    -- get the next unit to move
    local units = wesnoth.get_units({ side = ai.side })
    if (data['current_unit'] == nil or data['current_unit'] >= #units) then
        data['current_unit'] = 1
    else
        data['current_unit'] = data['current_unit'] + 1
    end

    local unit = units[data['current_unit']]
    std_print('**next_unit*' .. unit.x .. ',' .. unit.y)

    -- waits and reads the input file until the counter increased and a new action is available
    -- nil checks cause right after file save the values get shortly nil
    local action, counter = wesnoth.dofile(wml.variables.ai_input_path)
    while (counter == nil or action == nil or counter <= wml.variables.ai_round_counter) do
        action, counter = wesnoth.dofile(wml.variables.ai_input_path)

        -- show how many nils we get
        -- if (action == nil or counter == nil) then
        --     std_print(action)
        --     std_print(counter)
        -- end
    end
    wml.variables.ai_round_counter = counter

    -- save the unit and action, so the execution can use them
    data['next_move'] = {unit = unit, action = action}

    return 999990
end

-- executes the selected action for the current unit and prints the resulting game state
-- https://wiki.wesnoth.org/Creating_Custom_AIs#Execution_Function
-- https://wiki.wesnoth.org/LuaAI#ai_Table_Functions_for_Executing_AI_Moves
function ca_action_manager:execution(cfg, data)
    local next_move = data['next_move']
    local valid_move = true

    if (next_move.action > 0) then
        local x, y = calculate_new_position(next_move.unit, next_move.action)
    
        -- checks if the calculated x, y position is valid
        if (check_position_is_on_map(x, y) and check_unit_move(next_move.unit, x, y)) then
            -- std_print('Moving unit to (' .. x .. ',' .. y .. ')')
            ai.move(next_move.unit, x, y)
        else
            -- std_print('Cant move unit to (' .. x .. ',' .. y .. ')')
            valid_move = false
        end
    end

    -- sets the available moves of the current unit to zero (a needed game state change, so this CA doesn't get blacklisted)
    ai.stopunit_moves(next_move.unit)
    -- prints out the resulting game state and if the selected move was valid
    stdout_observation(valid_move)
end

-- checks if the given move for the given unit is valid
-- https://wiki.wesnoth.org/LuaAI#ai_Table_Functions_for_Executing_AI_Moves
function check_unit_move(unit, x, y)
    local gamestate = ai.check_move(unit, x, y)
    return gamestate.ok == true and gamestate.status == 0 and gamestate.result == 'action_result::AI_ACTION_SUCCESS'
end

-- checks if the given position is on the map
function check_position_is_on_map(x, y)
    local map_width, map_height, map_border = wesnoth.get_map_size()
    
    return ((x > 0) 
    and (y > 0) 
    and (x <= map_width) 
    and (y <= map_height))
end

-- cube coordinates to vertical odd offset coordinates
function cube_to_oddq(x, y, z)
    local col = x
    local row = z + (x - (x % 2)) / 2
    return col, row
end

-- cube coordinates to vertical even offset coordinates
function cube_to_evenq(x, y, z)
    local col = x
    local row = z + (x + (x % 2)) / 2
    return col, row
end

-- vertical odd offset coordinates to cube coordinates
function oddq_to_cube(col, row)
    local x = col
    local z = row - (col - (col % 2)) / 2
    local y = - x - z
    return x, y, z
end

-- vertical even offset coordinates to cube coordinates
function evenq_to_cube(col, row)
    local x = col
    local z = row - (col + (col % 2)) / 2
    local y = - x - z
    return x, y, z
end

-- https://arxiv.org/pdf/1803.02108.pdf
-- https://www.redblobgames.com/grids/hexagons/#conversions
-- range = 0 -> 1 action -> 1 possible
-- range = 1 -> 7 actions -> 7 possible
-- range = 2 -> 19 actions -> 13 possible
-- range = 3 -> 37 actions -> 19 possible
-- range = 4 -> 61 actions -> 25 possible
-- range = 5 -> 91 actions -> 31 possible
function calculate_new_position(unit, action)
    action = action - 1
    local index = action % 6
    local range = math.floor(action / 6 + 1)

    local x = 0
    local y = 0
    local z = 0

    -- calculating the new position offset in cube coordinate system
    if (index ==  0) then
        y = y + range
        z = z - range
    elseif (index == 1) then
        x = x + range
        z = z - range
    elseif (index == 2) then
        x = x + range
        y = y - range
    elseif (index == 3) then
        y = y - range
        z = z + range
    elseif (index == 4) then
        x = x - range
        z = z + range
    elseif (index == 5) then
        x = x - range
        y = y + range
    end

    -- transform cube coordinates to vertical offset coordinates
    local col, row
    if (unit.x % 2 == 0) then
        col, row = cube_to_evenq(x, y, z)
    else
        col, row = cube_to_oddq(x, y, z)
    end
    return unit.x + col, unit.y + row
end

-- convert the action index into a x, y position for the given unit
-- works only for action index 1 to 6
function calculate_new_position_old(unit, action)
    if (action == 1) then
        return unit.x, unit.y - 1
    elseif (action == 2) then
        if (unit.x % 2 == 1) then
            return unit.x + 1, unit.y - 1 
        else
            return unit.x + 1, unit.y
        end
    elseif (action == 3) then
        if (unit.x % 2 == 1) then
            return unit.x + 1, unit.y
        else
            return unit.x + 1, unit.y + 1
        end
    elseif (action == 4) then
        return unit.x, unit.y + 1
    elseif (action == 5) then
        if (unit.x % 2 == 1) then
            return unit.x - 1, unit.y
        else
            return unit.x - 1, unit.y + 1
        end
    elseif (action == 6) then
        if (unit.x % 2 == 1) then
            return unit.x - 1, unit.y - 1
        else
            return unit.x - 1, unit.y
        end
    end
end

-- prints out all relevant information of the current game state
function stdout_observation(action_success)
    local map_width, map_height, map_border = wesnoth.get_map_size()
    local village_count = #(wesnoth.get_villages({}))
    local tod = wesnoth.get_time_of_day()
    local side = wesnoth.sides[ai.side]
    local turn_max = wesnoth.game_config.last_turn
    local finished = wesnoth.current.turn == turn_max and all_units_moved()
    local own_units_count = #(wesnoth.get_units({ side = ai.side}))

    std_print('***start***')
    std_print('**map_size*(' .. map_width .. ',' .. map_height .. ',' .. map_border .. ')')
    std_print('**turn*' .. wesnoth.current.turn)
    std_print('**turn_max*' .. turn_max)
    std_print('**finished*' .. tostring(finished))
    std_print('**side*' .. ai.side)
    std_print('**tod*' .. tod.id)
    std_print('**valid_move*' .. tostring(action_success))
    std_print('**gold*'.. side.gold)
    std_print('**all_villages*'.. village_count)
    std_print('**own_villages*'.. side.num_villages)
    std_print('**own_units*'.. own_units_count)

    stdout_map(map_width, map_height)

    std_print('***end***')
end

-- prints out the current map of the current players view
-- ownerships are always positiv for units of the current player 
-- and negativ for enemy units
function stdout_map(map_width, map_height)
    local locations = wesnoth.get_locations({include_borders = false})
    local villages = wesnoth.get_villages({})
    local units = wesnoth.get_units({})

    local map = {}
    for i=1,map_width do
        map[i] = {}
        for j=1,map_height do
            map[i][j] = {}
        end
    end

    for index, location in ipairs(locations) do
        local terrain = wesnoth.get_terrain(location[1], location[2])
        -- local terrain_info = wesnoth.get_terrain_info(terrain)

        -- [0]: terrain info, [1]: village owner, [2]: unit owner
        map[location[1]][location[2]] = {terrain, 0, 0}
    end

    for index, village in ipairs(villages) do
        local village_owner = wesnoth.get_village_owner(village[1], village[2])
        if (village_owner) then
            local location = map[village[1]][village[2]]
            location[2] = base_own_side_number(ai.side, village_owner)
            map[village[1]][village[2]] = location
        end
    end

    for index, unit in ipairs(units) do
        local location = map[unit.x][unit.y]
        location[3] = base_own_side_number(ai.side, unit.side)
        map[unit.x][unit.y] = location
    end

    for x, row in ipairs(map) do
        for y, location in ipairs(row) do
            std_print(string.format('**map*%d,%d,%s,%s,%s', x, y, location[1], location[2], location[3]))
        end
    end
end

-- checks if all units of the current player have been moved
function all_units_moved()
    local units = wesnoth.get_units({ side = ai.side })
    local count = 0

    for index, unit in ipairs(units) do
        count = count + unit.moves
    end

    return count == 0
end

-- re setting owner ids, so current player has always index 1
-- and enemies have -1
-- needed for map mirroring
function base_own_side_number(own_side, side)
    if (side == own_side) then
        return 1
    else 
        return -1
    end
end

-- generate random input file name and wait for its creation
function init_and_wait_for_input_file_creation() 
    local uuid = uuid()
    wml.variables.ai_input_path = "~/input/" .. uuid .. ".lua"

    std_print('**inputPath*' .. wml.variables.ai_input_path)

    while (not wesnoth.have_file(wml.variables.ai_input_path)) do
    end
end

-- generates a random uuid
-- https://gist.github.com/jrus/3197011
function uuid()
    local template ='xxxxxxxxxx'
    return string.gsub(template, '[xy]', function (c)
        local v = (c == 'x') and math.random(0, 0xf) or math.random(8, 0xb)
        return string.format('%x', v)
    end)
end


return ca_action_manager