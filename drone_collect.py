"""
Keyboard-controlled drone demo collector for ProjectAirSim.

Controls
--------
  W / S     fly North / South
  A / D     fly West  / East
  Q / E     fly Up    / Down  (Q = up = negative NED-Down)
  P         start recording episode
  M         stop recording and save to HDF5
  Esc       land and quit

HDF5 layout (per episode)
-------------------------
  observations/
      images/scene  (T, H, W, 3)  uint8  BGR
      position      (T, 3)         float64  [x, y, z] NED
      velocity      (T, 3)         float64  [vx, vy, vz] NED
      orientation   (T, 4)         float64  [w, x, y, z] quaternion
  action            (T, 3)         float64  [v_north, v_east, v_down]
  tm                (T, 1)         float64  step duration (s)
"""

import asyncio
import datetime
import os
import sys
import threading
import time

import cv2
import h5py
import numpy as np
import projectairsim
from projectairsim.utils import unpack_image

try:
    from pynput import keyboard
    from pynput.keyboard import Key, KeyCode
except ImportError:
    print("Error: pynput is required.  Run: pip install pynput")
    sys.exit(1)

# ── Configuration ────────────────────────────────────────────────────────────
_SIM_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "client", "python", "example_user_scripts", "sim_config",
)
_SCENE_NAME = "scene_basic_drone.jsonc"
SPEED       = 3.0    # m/s
LOOP_HZ     = 10
SAVE_DIR    = "./data/drone_demos"
IMG_H, IMG_W = 480, 640

# ── Keyboard state ───────────────────────────────────────────────────────────
_keys_held    = set()
_keys_lock    = threading.Lock()
_record_start = threading.Event()   # edge-trigger on P
_record_save  = threading.Event()   # edge-trigger on M
_quit_flag    = threading.Event()
_p_down = _m_down = False

def _on_press(key):
    global _p_down, _m_down
    with _keys_lock:
        _keys_held.add(key)
    if key == KeyCode.from_char('p') and not _p_down:
        _p_down = True
        _record_start.set()
    if key == KeyCode.from_char('m') and not _m_down:
        _m_down = True
        _record_save.set()
    if key == Key.esc:
        _quit_flag.set()
        return False

def _on_release(key):
    global _p_down, _m_down
    with _keys_lock:
        _keys_held.discard(key)
    if key == KeyCode.from_char('p'): _p_down = False
    if key == KeyCode.from_char('m'): _m_down = False

def _held(char: str) -> bool:
    with _keys_lock:
        return KeyCode.from_char(char) in _keys_held

# ── Camera image buffer ──────────────────────────────────────────────────────
_img_lock   = threading.Lock()
_latest_img = None   # numpy BGR (H, W, 3)

def _image_cb(_, msg):
    global _latest_img
    img = unpack_image(msg)
    if img is not None:
        with _img_lock:
            _latest_img = img.copy()

def _grab_frame() -> np.ndarray:
    with _img_lock:
        if _latest_img is not None:
            return _latest_img.copy()
    return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

# ── Data recorder ────────────────────────────────────────────────────────────
class DataRecorder:
    def __init__(self):
        self.reset()

    def reset(self):
        self._buf = {k: [] for k in
                     ("pos", "vel", "ori", "img", "action", "tm")}

    @property
    def n_steps(self):
        return len(self._buf["action"])

    def record(self, pos, vel, ori, img, action, dt):
        self._buf["pos"].append(pos)
        self._buf["vel"].append(vel)
        self._buf["ori"].append(ori)
        self._buf["img"].append(img)
        self._buf["action"].append(action)
        self._buf["tm"].append([dt])

    def save(self, save_dir: str):
        T = self.n_steps
        if T == 0:
            print("  [WARN] no data — nothing saved")
            return
        os.makedirs(save_dir, exist_ok=True)
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(save_dir, f"episode_{ts}.hdf5")

        imgs = np.stack(self._buf["img"]).astype(np.uint8)  # (T, H, W, 3)
        with h5py.File(path, "w") as f:
            obs     = f.create_group("observations")
            img_grp = obs.create_group("images")
            img_grp.create_dataset("scene", data=imgs,
                                   chunks=(1, *imgs.shape[1:]))
            obs.create_dataset("position",    data=np.array(self._buf["pos"], dtype=np.float64))
            obs.create_dataset("velocity",    data=np.array(self._buf["vel"], dtype=np.float64))
            obs.create_dataset("orientation", data=np.array(self._buf["ori"], dtype=np.float64))
            f.create_dataset("action", data=np.array(self._buf["action"], dtype=np.float64))
            f.create_dataset("tm",     data=np.array(self._buf["tm"],     dtype=np.float64))

        print(f"  [OK] {T} steps saved → {path}")

# ── Main ─────────────────────────────────────────────────────────────────────
async def main():
    client = projectairsim.ProjectAirSimClient()
    client.connect()
    world  = projectairsim.World(client, _SCENE_NAME,
                                 sim_config_path=_SIM_CONFIG_PATH,
                                 delay_after_load_sec=2)
    drone  = projectairsim.Drone(client, world, "Drone1")

    # Subscribe to DownCamera before arming so frames start arriving immediately
    client.subscribe(drone.sensors["DownCamera"]["scene_camera"], _image_cb)

    drone.enable_api_control()
    drone.arm()
    print("Taking off…")
    await (await drone.takeoff_async())
    print("Airborne!\n")

    recorder     = DataRecorder()
    is_recording = False

    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()

    print("Controls:")
    print("  W/S  North/South    A/D  West/East    Q/E  Up/Down")
    print("  P    Start recording episode")
    print("  M    Save episode and stop recording")
    print("  Esc  Land and quit\n")

    dt     = 1.0 / LOOP_HZ
    t_prev = time.time()

    try:
        while not _quit_flag.is_set():
            t0        = time.time()
            actual_dt = t0 - t_prev
            t_prev    = t0

            # Recording control (edge-triggered)
            if _record_start.is_set():
                _record_start.clear()
                if not is_recording:
                    recorder.reset()
                    is_recording = True
                    print("  ● Recording started  (press M to save)")

            if _record_save.is_set():
                _record_save.clear()
                if is_recording:
                    is_recording = False
                    print(f"  ■ {recorder.n_steps} steps — saving…")
                    recorder.save(SAVE_DIR)

            # Velocity command from keyboard
            vn = ve = vd = 0.0
            if _held('w'): vn += SPEED
            if _held('s'): vn -= SPEED
            if _held('d'): ve += SPEED
            if _held('a'): ve -= SPEED
            if _held('q'): vd -= SPEED   # Q = up = negative NED-Down
            if _held('e'): vd += SPEED
            action = [vn, ve, vd]

            await drone.move_by_velocity_async(
                v_north=vn, v_east=ve, v_down=vd, duration=dt * 2
            )

            # Collect state + image
            img = _grab_frame()
            if is_recording:
                s     = drone.get_ground_truth_kinematics()
                pos_d = s["pose"]["position"]
                vel_d = s["twist"]["linear"]
                ori_d = s["pose"]["orientation"]
                recorder.record(
                    pos=[pos_d["x"], pos_d["y"], pos_d["z"]],
                    vel=[vel_d["x"], vel_d["y"], vel_d["z"]],
                    ori=[ori_d["w"], ori_d["x"], ori_d["y"], ori_d["z"]],
                    img=img,
                    action=action,
                    dt=actual_dt,
                )

            # Display with recording indicator
            disp  = img.copy()
            label = f"REC ● {recorder.n_steps} steps" if is_recording else "IDLE"
            color = (0, 0, 255) if is_recording else (180, 180, 180)
            cv2.putText(disp, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("DownCamera", disp)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            # Maintain loop rate
            elapsed = time.time() - t0
            if dt - elapsed > 0:
                await asyncio.sleep(dt - elapsed)

    finally:
        listener.stop()
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

if __name__ == "__main__":
    asyncio.run(main())
