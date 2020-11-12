----- CA: Manual move (max_score: 999980) -----

local ca_manual_move = {}

function ca_manual_move:evaluation()
    local start_time = wesnoth.get_time_stamp()
    -- std_print('*log*manual_move_eva' .. start_time)

    local timeout_ms = 500 -- milli-seconds
    --std_print('Start time:', start_time)

    local old_turn = wml.variables.manual_ai_turn
    local old_id = wml.variables.manual_ai_id
    local old_x = wml.variables.manual_ai_x
    local old_y = wml.variables.manual_ai_y
    local id, x, y = wesnoth.dofile("~/add-ons/ARL-Addon/lua/manual_input_" .. ai.side .. ".lua")

    local turn = wesnoth.current.turn

    if (old_turn and old_turn == turn) then
        local time = wesnoth.get_time_stamp()
        std_print('*log*manual_move_xxx' .. time)
        return 0
    end

    while (id == old_id) and (x == old_x) and (y == old_y) and (wesnoth.get_time_stamp() < start_time + timeout_ms) do
    -- while (id == old_id) and (x == old_x) and (y == old_y) do
        id, x, y = wesnoth.dofile("~/add-ons/ARL-Addon/lua/manual_input_" .. ai.side .. ".lua")
    end

    if (id == old_id) and (x == old_x) and (y == old_y) then
        std_print('manual move CA has timed out')

        local time = wesnoth.get_time_stamp()
        std_print('*log*manual_move_eva_tout' .. time)
        return 0
    else
        local time = wesnoth.get_time_stamp()
        std_print('*log*manual_move_eva' .. time)
        return 999980
    end
end

function ca_manual_move:execution(cfg, data)
    local start_time = wesnoth.get_time_stamp()
    std_print('*log*manual_move_exe' .. start_time)

    local id, x, y = wesnoth.dofile("~/add-ons/ARL-Addon/lua/manual_input_" .. ai.side .. ".lua")
    std_print('move ' .. id .. ' --> ' .. x .. ',' .. y)

    local unit = wesnoth.get_unit(id)

    pcall(move_unit, unit, x, y)

    wml.variables.manual_ai_id = id
    wml.variables.manual_ai_x = x
    wml.variables.manual_ai_y = y
    wml.variables.manual_ai_turn = wesnoth.current.turn
end

function move_unit(unit, x, y)
    ai.move(unit, x, y)
end

return ca_manual_move
