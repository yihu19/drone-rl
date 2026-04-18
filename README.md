# Drone-RL

Train_ppo.py for training drone with stablebaseline ppo


## Keyboard control
Controlling the drone using keyboard
```bash
python drone_keyboard.py
```

## Dataset Collection
Collect the dataset using keyboard, saving the state and image into hdf5 format
```bash
python drone_collect.py
```

Reply and check episodes of dataset
```bash
python drone_reply.py
```


## Behavior Cloning

This part is for learning the policy from dataset.

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

Eval the vision policy:
```bash
python bc_eval.py --ckpt ./bc_checkpoints/bc_vision_best.pt
```

Or eval the state policy:
```bash
python bc_eval.py --ckpt ./bc_checkpoints/bc_state_best.pt
```

## Plan:

RL (training from scratch, but with human real-time correction) training pipeline