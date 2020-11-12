# Benchmarking Adversarial Resilience Learning using the example of AI development for Battle for Wesnoth

In this project an [OpenAI Gym](https://gym.openai.com/) like environment for the game [Battle for Wesnoth](https://www.wesnoth.org/) was developed. In this environment the benefits of Multi-Agent Reinforcement Learning and specially the concepts of [Adversarial Resilience Learning](https://arxiv.org/abs/1811.06447) have been tested. The results show that Multi-Agent Reinforcement Learning creates more robust and sometimes also quicker AIs than learning the agent alone. 

The book "Deep Reinforcement Learning Hands-On" from Maxim Lapan was used as template for the Reinforcement Learning code. More in detail the library [PTAN](https://github.com/Shmuma/ptan) and the [code of the book](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On) was used. 

Based on the [ideas and discussion](https://forums.wesnoth.org/viewtopic.php?f=10\&t=51061) in the game forum an external control of the AI in Battle for Wesnoth was created. The AI is coded in Lua as Addon for the game ([PythonAddon](PythonAddon)) and gets controlled by external Python ([PythonController](PythonController)). In Python the game is executed as subprocess with the following command. It does enough runs of the imaginary scenario "sce42". 

```wesnoth --multiplayer --nogui --nosound --multiplayer-repeat 1000000 --scenario sce42```

The communication between Python and Lua is implemented as suggested in the forum. The Lua code returns data (mostly the observation data of the environment) through the standard output on the console of the subprocess. This output is piped into Python and is processed there for the AI training. The data, which is sent to the game, (mostly the selected actions of the Agents) is written to a file, which is read by the Lua code. 

## Local setup

First install the game Battle for Wesnoth, add the Addon for external control and install the Python environment. This setup is not needed for docker. 

### Install game

Install prebuild wesnoth package (most times not the latest developer version)

```apt-get install wesnoth```

Or build and install latest version locally. Download from https://www.wesnoth.org/#download and follow instructions in INSTALL.md from the download. 

### Install game addon

Get the path of userdata for BfW with ```wesnoth --userdata-path```. In the following replace ... by this path. 

Add the folder /input to the folder .../data/

Copy or mount the PythonAddon dictonary to .../data/add-ons/

### Install Python environment and requirements

Install Python 3.7 and make a virtual environment in the [PythonController](PythonController) folder. 

```python3.7 -m venv env```

Activate the environment. 

```source env/bin/activate```

Install the OpenAI Gym for Battle for Wesnoth. 

```pip3 install -e ./gym-bfw```

Install all other requirements. 

```pip3 install -r requirements.txt```

### Add replay saving in multiplayer games

There can be added some code to automatically save replays of the games played by the AIs. As the AIs play without a GUI, after training you can watch the replays to get observe the plays made by the AIs. 

Add to wesnoth-1.14.13/src/game_initialization/multiplayer.cpp following header 

```c++
#include "savegame.hpp"
```

and modify the for loop around line 796 with the last two rows. 

```c++ 
for(unsigned int i = 0; i < repeat; i++){
    saved_game state_copy(state);
    campaign_controller controller(state_copy, game_config_manager::get()->terrain_types());
    controller.play_game();
    
    // added to automatically save replays
    savegame::replay_savegame save(state_copy, preferences::save_compression_format());
    save.save_game_automatic(false, "my_filename" + std::to_string(i));
}
```

Self compile afterwards.

## Run the Controller and train AIs

This project can be run in two ways - as docker containers or directly with Python. For both options there are different run configurations given. If you run a test scenario with the usage of different reward functions for the attacking and the defending agent, you have to uncommend the lines 250-252 in [bfw_env.py](PythonController/gym-bfw/gym_bfw/envs/bfw_env.py). 

### Docker

There is a [Dockerfile ](Dockerfile) for building a container with Battle for Wesnoth, the PythonAddon and the PythonController installed. The [docker-compose.yml](docker-compose.yml) contains different run configurations. It is important to set ROOT_PATH in [.env](.env), cause it is used in the docker-compose.yml for the output of the container (TensorBoard data, trained neural networks). 

Some docker commands used during testing on a NVIDIA DGX-1: 

Current GPU resource usage: 

```nvidia-smi```

Show resource usage of container X:

```docker stats X```

Show processes fo container X:

```docker top X```

Build the container:
```docker build --tag masterthesis-bfw-1023 --build-arg USER_ID=10030 .```

Run docker-compose and rebuild all images:

```docker-compose up -d --build```

Run contaiter X from docker-compose and remove it afterwards:

```docker-compose run --rm -d X```

Delete all created in docker-compose:

```docker-compose down```

Tail log of container X:

```docker logs -f X```

Run nvidia-smi in container X:

```docker exec X nvidia-smi```

### Locally (Configuration for Visual Studio Code)

The project can also be run inside of Visual Studio Code with configured examples in [launch.json](PythonController/.vscode/launch.json). Therefor open Visual Studio Code in the [PythonController](PythonController) folder and select a run configuration. 

### Run only the game and the Lua code

If you want to control the AI in the game manually, you can start the game like the PythonController would do. Be aware that you have to manually edit the input file for the actions, cause you are now the agent yourself. 

```wesnoth --multiplayer --scenario multi_side_ai_3units```

## Results

The are multiple training and test scenarios. Here are shown some examples of test scenarios for the combinations of singleplayers and multiplayers. 

Singleplayer:

![SingleplayerVideo](Videos/singleplayer.gif)


Singleplayer vs Singleplayer:

![SingleplayerVsSingleplayerVideo](Videos/singleplayer-vs-singleplayer.gif)


Multiplayer vs Multiplayer:

![MultiplayerVsMultiplayerVideo](Videos/multiplayer-vs-multiplayer.gif)


Multiplayer (blue units) vs Singleplayer (red units):

![MultiplayerVsSingleplayerVideo](Videos/singleplayer-vs-multiplayer.gif)
