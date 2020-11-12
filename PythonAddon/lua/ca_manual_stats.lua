-- https://wiki.wesnoth.org/Luaai#Lua_AI_Functions
-- https://wiki.wesnoth.org/LuaWML
-- https://wiki.wesnoth.org/LuaWML/Pathfinder

----- CA: Stats at beginning of turn (max_score: 999990) -----
-- This will be blacklisted after first execution each turn

local ca_manual_stats = {}
math.randomseed( os.time() )

function ca_manual_stats:evaluation()
    local start_time = wesnoth.get_time_stamp()
    std_print('*log*manual_stats_eva' .. start_time)
    return 999990
end

function ca_manual_stats:execution(cfg, data)
    local start_time = wesnoth.get_time_stamp()
    std_print('*log*manual_stats_exe' .. start_time)

    local map_width, map_height, map_border = wesnoth.get_map_size()
    local tod = wesnoth.get_time_of_day()
    local units = wesnoth.get_units({})
    local locations = wesnoth.get_locations({include_borders = false})
    local nov = #wesnoth.get_villages({ owner_side = ai.side })

    std_print('***start***')
    std_print('**map_size*(' .. map_width .. ',' .. map_height .. ')')
    std_print('**turn*' .. wesnoth.current.turn)
    std_print('**side*' .. ai.side)
    std_print('**tod*' .. tod.id)
    std_print('**nov*' .. nov)

    for index, location in ipairs(locations) do
        local terrain = wesnoth.get_terrain(location[1], location[2])
        local terrain_info = wesnoth.get_terrain_info(terrain)

        local unit = wesnoth.get_unit(location[1], location[2])
        local unit_side = 0
        if unit ~= nil then
            unit_side = unit.side
        end

        std_print(string.format('**map*%d,%d,%s,%s', location[1], location[2], terrain, unit_side))
    end

    std_print('***end***')

    local id, x, y = wesnoth.dofile("~/add-ons/ARL-Addon/lua/manual_input_" .. ai.side .. ".lua")
    wml.variables.manual_ai_id = id
    wml.variables.manual_ai_x = x
    wml.variables.manual_ai_y = y
end

return ca_manual_stats
