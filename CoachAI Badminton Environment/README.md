## Requirement
- Python 
    ``` 
    python 3.8.10
    ```
- This project is relied on DyMF, ShuttleNet, StrategicEnvironment. Please satisfy they're requirements first.
- You needs to additionally install the following modules: 
    ```
    PySide6
    gym 
    ```
- Note: make sure your matplotlib is using the latest version, older version does not work with pySide6 (at least I got error when using matplotlib 3.5.1, now I upgrade to 3.7.0).

## How to run
1. Training three models: ShuttleNet, DyMF and BC
    1. DyMF: run the code in MovementForecasting/train.py
        - for details, please refer to the [DyMF github](https://github.com/wywyWang/CoachAI-Projects/tree/main/Movement%20Forecasting).
        - For simple experiment, you can run
            ```
            ./train.py --preprocessed_data_path demo.csv
            ```
            inside MovementForecasting folder. `demo.csv` is a one-set data only for showcase, for complete training, you can replace the demo data with ShuttleSet, which is an open-source dataset we used.
    2. ShuttleNet: run the code in StrokeForecasting/train.py
        - for details, please refer to the [ShuttleNet github](https://github.com/wywyWang/CoachAI-Projects/tree/main/Stroke%20Forecasting).
        - For simple experiment, you can run
            ```
            ./train.py
            ```
            inside StrokeForecasting folder. For complete training, you can also replace the demo data, and change the `config['filename']` init setting in `badmintoncleaner.py` line `161`
    - Note that for both models, we add GMM to them for better performance.
    3. BC: run the code in First2Ball/train.py
        - It will training use `demo.csv`. `demo.csv` is a one-set data only for showcase, for complete training, you can replace the demo data with ShuttleSet, which is an open-source dataset we used.
2. Prepare you own model that is going to be trained.
    - Your model must inherit form the `RLModel` class
    - Either import it through gui or no-gui
        - For no gui version, coded directly into generate-nogui.py (recommended)
        - For gui version, you need to pack your model into a pt file and import using gui file dialog.
    - The output csv file is the log of two model fight each other, you can observe how two model performed.
3. After finish training, save the checkpoint and turn off the training in your code.
4. Load the model again, gui and non-gui can also be used. The output csv file will be visualized later.
5. Visualize. Run mainwindow.py to start gui program. 

## API Usage
- Our environment is based on the well-known gym environment.
- The example usage is as following:<br/>
    similar version of `run_CoachAI.py`, `agent` is a custom aget that inherent from `RLModel`
    ```python
    from CoachAI import badminton_env
    
    episodes = 1000
    env = badminton_env(episodes, side=1, opponent="Anthony Sinisuka GINTING")
    
    for episode in range(1, episodes+1):
        state, launch = env.reset()
        done = False
    
        while not done :
            action = agent.train(state, launch)
            state, reward, done, launch = env.step(action, launch)
    
            print(f"State: {state}")
            print(f"Action: {action}")
    
    env.save()
    ```  
    - badminton_env() to init to environment, you can specify you episode, side and opponent here
        - `episode`: the rally to run
        - `side`: custom model is player A or B, only affect the initial launcher
        - `opponent`: the opponent from realistic agent
    - env.reset() to reset the enviroment, it will return `state` and `launch`, where `state` is the initial state, and `launch` indicate current ball is launch ball or not.
        - `state`: current state, which is initial state if `launch` is True; if `launch` is False, the state is the return state after realistic agent perform launch.
        - `launch`: if custom model need to perform launch ball or not
    - env.step() to play a turn with realistic opponent, it will return `state`, `reward`, `done`, and `launch`
        - `state`: current state
        - `reward`: reward of previous step
        - `done`: whether current rally is done
        - `launch`: whether next ball is launch or not, `True` only when current rally is end
    - env.save() to output the playing record to csv, you can also pass the filename as parameter.



## Structure
### Outline
- This project is based on three model:
    1. DyMF([github](https://github.com/wywyWang/CoachAI-Projects/tree/main/Movement%20Forecasting))
    2. ShuttleNet([github](https://github.com/wywyWang/CoachAI-Projects/tree/main/Stroke%20Forecasting))
    3. Strategic Environment([github](https://github.com/wywyWang/CoachAI-Projects/tree/main/Strategic%20Environment))
- The whole flow is:
    1. ShuttleNet, DyMF, BC composes to Realistic Opponent, build in the RL Enviroment.
    2. The model trained with the RL Enviornment will generate a csv log file.
    3. Use Strategic Environment as backend to stimulate this csv file, the front end is Qt, the statistic graph is rendered with matplotlib

### File Structure
- the following is the part of our file structure, we list out the file which is important to our project.
    - ```bash
        CoachAI
        ├─GUI
        │  ├─field.py
        │  ├─*.ui
        │  ├─*_ui.py
        │  └─*Widget.py
        ├─MovementForecasting
        │  └─...
        ├─StrategicEnvironment
        │  └─...
        ├─StrokeForecasting
        │  └─...
        ├─First2Ball
        │  ├─BC.py
        │  └─train.py
        ├─Utils
        │  └─...
        ├─DyMFAgent.py
        ├─First2Ball.py
        ├─generate_nogui.py
        ├─generateWidgetCtrl.py
        ├─mainwindow.py
        ├─RLEnvironment.py
        ├─RLModel.py
        ├─ShuttleNetAgent.py
        ├─stimulateTabCtrl.py
        ├─SuperviseAgent.py
        ├─CoachAI.py
        ├─run_CoachAI.py
        ├─ppo.py
        ├─ddpg.py
        └─a2c.py
        ```
    - The `StrokeForecasting` folder contains the code of ShuttleNet
    - The `MovementForesting` folder contains the code of DyMF
    - The `First2Ball` folder contains the code for BC
    - The `StrategicEnvironment` folder contains the code the stimulation environment of visualization.
    - The `Utils` folder contains some small tools to statistic or precessing dataset.
    - The `GUI` folder contains the utils of gui needed file.
        - The `*.ui` and `*_ui.py` is generated by Qt Creator(a gui form editor). If you need to change the gui appearance, you can open these files with Qt Creator. Also, modify them only with Qt Creator.
        - `*Widget.py`: Its only a wrapper of the file auto generated by Qt Creator, modify it may not need. Each *Widget.py will be imported by a *WidgetTabCtrl.py.
        - `field.py`: the main ctrl code of the field. You can control the players and ball moving by calling function in it.
    - `SuperviseAgent.py`, `DyMFAgent.py`, `ShuttleNetAgent.py`, `First2Ball.py` the code for realistic agents, for more details, view the [Realistic Opponent Structure](#realistic-opponent-structure) section below.
    - `*TabCtrl.py`: the controller of the each tab
    - `mainwindow.py`: the start code of gui
    - `RLEnvironment.py`: the gym base enviroment class
    - `RLModel.py`: the model template class
    - `ppo.py`, `ddpg.py`, `a2c.py`: The custom models as benchmark models.
### GUI Program Structure:
- `mainwindow.py`: the window frame of the program, it willcall the following files to init tabs
    - `stimulateTabCtrl.py`: control the behavior of visualization tab
        - `GUI/field.py`: control the display of the field shown on the left side
        - `StrategicEnvironment/`: the EnviornmentThread class will link the this environment
    - `generateWidgetCtrl.py`: control the behavior of generation tab
        - The structure in it is similar to nogui version, see below.
### Non-gui Program Structure:
- `run_CoachAI.py`: The sample usage of this environment, detail usageview the [api section](#api-usage)
    - `CoachAI.py` - The wrapped to two-models-based environment and our realistic opponents
        - `RLEnvironment.py`: the environment core of this RL enviroment, two agents will engage in a badminton match within this RL    environment.
        - `SuperviseAgent.py`: the realistic agent, include BC, ShuttleNet, DyMF in it, see [below](#realistic-opponent-structure) for detail structure.
            - `ShuttleNetAgent.py`
            - `DyMFAgent.py`
            - `First2Ball/BC.py`
    - your custom model, import mamually
- `generate-nogui.py`(deprecated): The combined version of `run_CoachAI.py` and `CoachAI.py`, need to choose two models, either realistic opponents or cutsom model.
### Realistic Opponent Structure:
- `SuperviseAgent.py`: the wrapper of ShuttleNet, DyMF andBC, it will call BC for first two balls and ShuttleNet andDyMF for other conditions. You can adjust the player andthe BC checkpoint here.
    - `ShuttleNetAgent.py`: the entry file of ShuttleNet. You can change the ShuttleNet checkpoint here. It will call into the ShuttleNet's own code(`StrokeForecasting/utils.py`) later.
    - `DyMFAgent.py`: the entry file of DyMFAgent. You can change the DyMF checkpoint here. It will call into the ShuttleNet's own code(`MovementForecasting/DyMF/runner.py`) later.
    - `First2Ball/BC.py`: the main code of BC, it is used to predict the first two ball.