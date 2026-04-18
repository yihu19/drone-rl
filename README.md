# Drone-RL


## Setup Instruction

The code is running based on [ProjectAirSim](https://github.com/iamaisim/ProjectAirSim/tree/main).

Please check the instruction for [pre-built binary environment](https://github.com/iamaisim/ProjectAirSim/blob/main/docs/development/use_prebuilt.md).


Please clone the [ProjectAirSim](https://github.com/iamaisim/ProjectAirSim/tree/main).
Then put this folder under ProjectAirSim

## Keyboard Control

Terminal-1: Run the simulator
```bash
./ProjectAirSim_Blocks/Linux/Blocks.sh
```

Terminal-2: Controlling the drone using keyboard
```bash
python drone_keyboard.py
```

## Dataset Collection

Terminal-1: Run the simulator
```bash
./ProjectAirSim_Blocks/Linux/Blocks.sh
```

Terminal-2: Collect the dataset using keyboard, saving the state and image into hdf5 format
```bash
python drone_collect.py
```

Reply and check episodes of dataset
```bash
python drone_reply.py
```


## Behavior Cloning

This part is for learning the policy from dataset collected by keyboard.

### Training

Train the vision policy
```bash
python bc_train.py \
  --data_dir ./data/drone_demos \
  --mode vision \
  --epochs 100 \
  --batch_size 64 \
  --lr 1e-3 \
  --log_dir ./bc_logs/vision \
  --save_dir ./bc_checkpoints
```
Or
train the state policy
```bash
python bc_train.py \
  --data_dir ./data/drone_demos \
  --mode state \
  --epochs 100 \
  --batch_size 64 \
  --lr 1e-3 \
  --log_dir ./bc_logs/state \
  --save_dir ./bc_checkpoints
```

### Eval

Terminal-1: Run the simulator
```bash
./ProjectAirSim_Blocks/Linux/Blocks.sh
```

Terminal-2: Eval the vision policy:
```bash
python bc_eval.py --ckpt ./bc_checkpoints/bc_vision_best.pt
```

Or eval the state policy:
```bash
python bc_eval.py --ckpt ./bc_checkpoints/bc_state_best.pt
```



## RL Training


### Training via via Stable_baselines3

Terminal-1: Run the simulator
```bash
./ProjectAirSim_Blocks/Linux/Blocks.sh
```

Using stable_baselines3 for training
```bash
python train_stablebaselines.py
```

Note: Here we use PPO, you can change it to other RL methods in stable_baselines3


### PPO Training

Terminal-1: Run the simulator
```bash
./ProjectAirSim_Blocks/Linux/Blocks.sh
```

Terminal-2: Using ppo for training
```bash
python PPO_training.py
```


### PPO Training with Human Correction

Terminal-1: Run the simulator
```bash
./ProjectAirSim_Blocks/Linux/Blocks.sh
```

Terminal-2: Using ppo for training
```bash
python PPO_training_HF.py --enable-keyboard
```
Or with stronger keyboard correction:
```bash
python PPO_training.py --enable-keyboard --keyboard-speed 10.0 --keyboard-mode add
```
Or if you want the keyboard to fully replace PPO whenever you press keys:
```bash
python PPO_training.py --enable-keyboard --keyboard-mode override
```

### Evaluation for PPO

Terminal-1: Run the simulator
```bash
./ProjectAirSim_Blocks/Linux/Blocks.sh
```

Terminal-2: run the evaluation
```bash
python PPO_eval.py --ckpt ./ppo_custom_checkpoints/ppo_actor.pt
```


## Drawn Path Following

Draw your path, then drone moves along the drawn path

Terminal-1: Run the simulator
```bash
./ProjectAirSim_Blocks/Linux/Blocks.sh
```

Terminal-2: run the evaluation
```bash
python draw_trajectory.py
```

