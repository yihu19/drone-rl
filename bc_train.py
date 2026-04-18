"""
Behavior Cloning for the ProjectAirSim drone.

Two modes (--mode flag):
  state   MLP on [position, velocity, orientation] → action  (default, fast)
  vision  CNN(image) + MLP(state) → action

Usage
-----
  python bc_train.py --data_dir ./data/drone_demos --mode state
  python bc_train.py --data_dir ./data/drone_demos --mode vision --epochs 50
"""

import argparse
import glob
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# ── Constants ────────────────────────────────────────────────────────────────
STATE_DIM  = 10   # pos(3) + vel(3) + ori(4)
ACTION_DIM = 3    # v_north, v_east, v_down
MAX_SPEED  = 5.0  # m/s  (matches drone_env action space)
IMG_SIZE   = 96   # resize to 96×96 for vision encoder

# ── Dataset ──────────────────────────────────────────────────────────────────
class DroneDataset(Dataset):
    """Loads all episodes from a directory of episode_*.hdf5 files."""

    def __init__(self, data_dir: str, use_images: bool = False):
        files = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
        if not files:
            raise FileNotFoundError(f"No episode_*.hdf5 files in {data_dir}")

        self.use_images = use_images
        self.states  = []
        self.actions = []
        self.images  = []   # only populated when use_images=True

        for path in files:
            with h5py.File(path, "r") as f:
                pos = f["observations/position"][()]       # (T, 3)
                vel = f["observations/velocity"][()]       # (T, 3)
                ori = f["observations/orientation"][()]    # (T, 4)
                act = f["action"][()]                      # (T, 3)
                state = np.concatenate([pos, vel, ori], axis=1)  # (T, 10)
                self.states.append(state.astype(np.float32))
                self.actions.append(act.astype(np.float32))

                if use_images:
                    imgs = f["observations/images/scene"][()]  # (T, H, W, 3)
                    self.images.append(imgs)

            print(f"  loaded {len(act):4d} steps  ← {os.path.basename(path)}")

        self.states  = np.concatenate(self.states,  axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        if use_images:
            self.images = np.concatenate(self.images, axis=0)
        print(f"  total {len(self.states)} timesteps")

        # Normalise actions to [-1, 1] for training stability
        self.actions_norm = (self.actions / MAX_SPEED).clip(-1, 1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state  = torch.from_numpy(self.states[idx])
        action = torch.from_numpy(self.actions_norm[idx])
        if not self.use_images:
            return state, action

        img = self.images[idx]                          # (H, W, 3) uint8
        img = torch.from_numpy(img).float() / 255.0    # (H, W, 3) in [0,1]
        img = img.permute(2, 0, 1)                     # (3, H, W)
        img = F.interpolate(img.unsqueeze(0),           # resize to IMG_SIZE
                            size=(IMG_SIZE, IMG_SIZE),
                            mode="bilinear",
                            align_corners=False).squeeze(0)
        return img, state, action

# ── Models ───────────────────────────────────────────────────────────────────
class StateBC(nn.Module):
    """MLP policy: state → action."""

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.net(state)   # in [-1, 1]; multiply by MAX_SPEED at inference


class VisionBC(nn.Module):
    """CNN image encoder + MLP state encoder → action."""

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        super().__init__()
        # Lightweight CNN: 96×96 → 6×6 feature map → 256-d
        self.img_enc = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),   # 96→23
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),  # 23→10
            nn.Conv2d(64, 128, 3, stride=2), nn.ReLU(), # 10→4
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
        )
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + 64, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, img, state):
        return self.head(torch.cat([self.img_enc(img),
                                    self.state_enc(state)], dim=1))

# ── Training ─────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}  |  Mode: {args.mode}\n")

    use_images = (args.mode == "vision")
    print("Loading dataset…")
    dataset = DroneDataset(args.data_dir, use_images=use_images)

    # 90 / 10 split
    val_n   = max(1, int(len(dataset) * 0.1))
    train_n = len(dataset) - val_n
    train_ds, val_ds = random_split(dataset, [train_n, val_n],
                                    generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)

    model = (VisionBC() if use_images else StateBC()).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    loss_fn   = nn.MSELoss()

    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    best_val = float("inf")
    ckpt_path = os.path.join(args.save_dir, f"bc_{args.mode}_best.pt")
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ── train ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            if use_images:
                img, state, action = [b.to(device) for b in batch]
                pred = model(img, state)
            else:
                state, action = [b.to(device) for b in batch]
                pred = model(state)

            loss = loss_fn(pred, action)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(state)

        train_loss /= train_n

        # ── validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                if use_images:
                    img, state, action = [b.to(device) for b in batch]
                    pred = model(img, state)
                else:
                    state, action = [b.to(device) for b in batch]
                    pred = model(state)
                val_loss += loss_fn(pred, action).item() * len(state)
        val_loss /= val_n

        scheduler.step()

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val",   val_loss,   epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.5f}  val={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": epoch,
                        "mode": args.mode,
                        "state_dict": model.state_dict()}, ckpt_path)
            print(f"  ✓ saved best model → {ckpt_path}")

    writer.close()
    print(f"\nTraining done.  Best val loss: {best_val:.5f}")
    print(f"Checkpoint: {ckpt_path}")

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="./data/drone_demos")
    parser.add_argument("--mode",       default="state", choices=["state", "vision"])
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--log_dir",    default="./bc_logs")
    parser.add_argument("--save_dir",   default="./bc_checkpoints")
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
