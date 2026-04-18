"""
Run a trained Behavior Cloning policy in ProjectAirSim.

Usage
-----
  python bc_eval.py --ckpt ./bc_checkpoints/bc_state_best.pt
  python bc_eval.py --ckpt ./bc_checkpoints/bc_vision_best.pt
"""

import argparse
import asyncio
import math
import os
import threading

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import projectairsim
from projectairsim.types import Pose, Vector3, Quaternion
from projectairsim.utils import unpack_image

from bc_train import StateBC, VisionBC, STATE_DIM, ACTION_DIM, MAX_SPEED, IMG_SIZE

# ── Sim config ────────────────────────────────────────────────────────────────
_SIM_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "client", "python", "example_user_scripts", "sim_config",
)
_SCENE_NAME = "scene_basic_drone.jsonc"

_yaw = math.radians(-45)
_SPAWN_POSE = Pose({
    "translation": Vector3({"x": -1.0, "y": 8.0, "z": -4.0}),
    "rotation": Quaternion({"w": math.cos(_yaw / 2), "x": 0.0,
                            "y": 0.0, "z": math.sin(_yaw / 2)}),
})

TARGET = np.array([9.0, 8.0, -9.0])   # same target as drone_env.py

# ── Image buffer (for vision mode) ───────────────────────────────────────────
_img_lock   = threading.Lock()
_latest_img = None

def _image_cb(_, msg):
    global _latest_img
    img = unpack_image(msg)
    if img is not None:
        with _img_lock:
            _latest_img = img.copy()

def _grab_frame():
    with _img_lock:
        if _latest_img is not None:
            return _latest_img.copy()
    return np.zeros((480, 640, 3), dtype=np.uint8)

# ── Policy inference ──────────────────────────────────────────────────────────
def load_policy(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    mode = ckpt["mode"]
    model = (VisionBC() if mode == "vision" else StateBC()).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded {mode} policy from {ckpt_path}  (epoch {ckpt['epoch']})")
    return model, mode

@torch.no_grad()
def predict(model, mode, state_np: np.ndarray, img_np=None, device="cpu"):
    state = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0).to(device)
    if mode == "vision" and img_np is not None:
        img = torch.from_numpy(img_np).float() / 255.0   # (H, W, 3)
        img = img.permute(2, 0, 1).unsqueeze(0).to(device)
        img = F.interpolate(img, size=(IMG_SIZE, IMG_SIZE),
                            mode="bilinear", align_corners=False)
        action_norm = model(img, state).cpu().numpy()[0]
    else:
        action_norm = model(state).cpu().numpy()[0]
    return action_norm * MAX_SPEED   # scale back to m/s

# ── Main eval loop ────────────────────────────────────────────────────────────
async def run_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mode = load_policy(args.ckpt, device)

    client = projectairsim.ProjectAirSimClient()
    client.connect()
    world  = projectairsim.World(client, _SCENE_NAME,
                                 sim_config_path=_SIM_CONFIG_PATH,
                                 delay_after_load_sec=2)
    drone  = projectairsim.Drone(client, world, "Drone1")

    if mode == "vision":
        client.subscribe(drone.sensors["DownCamera"]["scene_camera"], _image_cb)

    for episode in range(1, args.episodes + 1):
        print(f"\n── Episode {episode}/{args.episodes} ──")

        # Reset
        try:
            drone.disarm()
        except Exception:
            pass
        drone.set_pose(_SPAWN_POSE, True)
        drone.enable_api_control()
        drone.arm()
        await (await drone.takeoff_async())
        print("  Airborne — running BC policy")

        dt      = 1.0 / args.hz
        reached = False

        for step in range(args.max_steps):
            # Get state
            s     = drone.get_ground_truth_kinematics()
            pos_d = s["pose"]["position"]
            vel_d = s["twist"]["linear"]
            ori_d = s["pose"]["orientation"]
            pos   = np.array([pos_d["x"], pos_d["y"], pos_d["z"]])
            vel   = np.array([vel_d["x"], vel_d["y"], vel_d["z"]])
            ori   = np.array([ori_d["w"], ori_d["x"], ori_d["y"], ori_d["z"]])
            state = np.concatenate([pos, vel, ori])   # (10,)

            dist  = float(np.linalg.norm(TARGET - pos))

            # Policy inference
            img = _grab_frame() if mode == "vision" else None
            action = predict(model, mode, state, img, device)
            vn, ve, vd = float(action[0]), float(action[1]), float(action[2])

            await drone.move_by_velocity_async(
                v_north=vn, v_east=ve, v_down=vd, duration=dt * 2
            )

            # Display
            frame = _grab_frame()
            info  = (f"step {step+1}  dist={dist:.2f}m  "
                     f"v=({vn:.1f},{ve:.1f},{vd:.1f})")
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 0), 2)
            cv2.imshow("BC Policy", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                print("  Aborted.")
                break

            await asyncio.sleep(dt)

            if dist < 2.0:
                print(f"  ✅ Target reached in {step+1} steps!")
                reached = True
                break

        if not reached:
            print(f"  Did not reach target (final dist={dist:.2f}m)")

    cv2.destroyAllWindows()
    print("\nLanding…")
    try:
        await (await drone.land_async())
    except Exception:
        pass
    drone.disarm()
    drone.disable_api_control()
    client.disconnect()
    print("Done.")

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",      required=True,
                        help="Path to bc_state_best.pt or bc_vision_best.pt")
    parser.add_argument("--episodes",  type=int,   default=10)
    parser.add_argument("--max_steps", type=int,   default=300)
    parser.add_argument("--hz",        type=float, default=10.0)
    args = parser.parse_args()
    asyncio.run(run_eval(args))

if __name__ == "__main__":
    main()
