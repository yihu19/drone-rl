"""
Replay a recorded drone episode in ProjectAirSim.

Usage
-----
  python drone_replay.py                       # pick from ./data/drone_demos/
  python drone_replay.py --data_dir ./data/drone_demos
  python drone_replay.py --file ./data/drone_demos/episode_20260418_143000.hdf5
"""

import argparse
import asyncio
import glob
import math
import os

import cv2
import h5py
import numpy as np
import projectairsim
from projectairsim.types import Pose, Vector3, Quaternion

# ── Sim config (same as drone_collect.py) ────────────────────────────────────
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

# ── Episode selection ─────────────────────────────────────────────────────────
def list_episodes(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
    return files

def select_episode(data_dir: str, forced_file: str = None) -> str:
    if forced_file:
        if not os.path.isfile(forced_file):
            raise FileNotFoundError(f"File not found: {forced_file}")
        return forced_file

    files = list_episodes(data_dir)
    if not files:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {data_dir}")

    print("\nAvailable episodes:")
    print("-" * 60)
    for i, f in enumerate(files):
        with h5py.File(f, "r") as h:
            T = h["action"].shape[0]
        print(f"  {i+1:2d}.  {os.path.basename(f)}  ({T} steps)")
    print("-" * 60)

    while True:
        choice = input(f"Select episode [1-{len(files)}] or q to quit: ").strip()
        if choice.lower() == "q":
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return files[idx]
        except ValueError:
            pass
        print(f"  Enter a number between 1 and {len(files)}")

# ── Replay ────────────────────────────────────────────────────────────────────
async def replay(episode_path: str):
    # Load data
    with h5py.File(episode_path, "r") as f:
        actions = f["action"][()]                          # (T, 3)
        tm      = f["tm"][()]                              # (T, 1)
        images  = f["observations/images/scene"][()]      # (T, H, W, 3)
        pos_rec = f["observations/position"][()]          # (T, 3)

    T = len(actions)
    print(f"\nLoaded {T} steps from {os.path.basename(episode_path)}")

    # ── Preview recorded frames ───────────────────────────────────────────────
    print("Previewing recorded frames… press any key to advance, Q to skip.")
    for i, img in enumerate(images):
        disp = img.copy()
        cv2.putText(disp, f"Recorded {i+1}/{T}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        cv2.imshow("Recorded episode", disp)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q') or key == 27:
            break
    cv2.destroyAllWindows()

    # ── Connect to sim ────────────────────────────────────────────────────────
    print("\nConnecting to simulator…")
    client = projectairsim.ProjectAirSimClient()
    client.connect()
    world  = projectairsim.World(client, _SCENE_NAME,
                                 sim_config_path=_SIM_CONFIG_PATH,
                                 delay_after_load_sec=2)
    drone  = projectairsim.Drone(client, world, "Drone1")

    # Reset to spawn
    try:
        drone.disarm()
    except Exception:
        pass
    drone.set_pose(_SPAWN_POSE, True)
    drone.enable_api_control()
    drone.arm()

    print("Taking off…")
    await (await drone.takeoff_async())
    print(f"Replaying {T} steps — press Esc in the window to abort.\n")

    try:
        for i, (action, dt_arr) in enumerate(zip(actions, tm)):
            vn, ve, vd = float(action[0]), float(action[1]), float(action[2])
            dt = max(float(dt_arr[0]), 0.02)   # clamp to at least 20 ms

            # Send recorded velocity command
            await drone.move_by_velocity_async(
                v_north=vn, v_east=ve, v_down=vd, duration=dt * 2
            )

            # Side-by-side: recorded frame | live position text
            disp = images[i].copy()
            cv2.putText(disp, f"Step {i+1}/{T}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Replay", disp)
            if cv2.waitKey(1) & 0xFF == 27:
                print("  Aborted by user.")
                break

            await asyncio.sleep(dt)

        print("Replay complete.")

        # Compare final position with recording
        s = drone.get_ground_truth_kinematics()
        p = s["pose"]["position"]
        live_pos = np.array([p["x"], p["y"], p["z"]])
        rec_pos  = pos_rec[-1]
        diff     = np.linalg.norm(live_pos - rec_pos)
        print(f"  Recorded final pos : {rec_pos}")
        print(f"  Live final pos     : {live_pos}")
        print(f"  Position error     : {diff:.3f} m")

    finally:
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
    parser = argparse.ArgumentParser(description="Replay a drone episode in ProjectAirSim")
    parser.add_argument("--data_dir", default="./data/drone_demos",
                        help="Directory containing episode_*.hdf5 files")
    parser.add_argument("--file", default=None,
                        help="Path to a specific episode file (skips selection menu)")
    args = parser.parse_args()

    episode = select_episode(args.data_dir, args.file)
    if episode is None:
        return

    asyncio.run(replay(episode))

if __name__ == "__main__":
    main()
