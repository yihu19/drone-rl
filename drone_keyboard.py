"""
Keyboard control for ProjectAirSim drone (scene_basic_drone.jsonc).

Controls (hold keys):
  W / S   — fly North / South
  A / D   — fly West  / East
  Q / E   — fly Up    / Down
  Space   — hover (zero velocity)
  Escape  — land and quit
"""

import asyncio
import os
import sys
import threading

import projectairsim

try:
    from pynput import keyboard
    from pynput.keyboard import Key, KeyCode
except ImportError:
    print("Error: pynput is required.  Run: pip install pynput")
    sys.exit(1)

# --------------------------------------------------------------------------- #
_SIM_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "client", "python", "example_user_scripts", "sim_config",
)
_SCENE_NAME = "scene_basic_drone.jsonc"

SPEED      = 3.0   # m/s per axis
LOOP_HZ    = 20    # control loop frequency

# --------------------------------------------------------------------------- #
keys_held: set = set()
quit_flag = threading.Event()

def _on_press(key):
    keys_held.add(key)

def _on_release(key):
    keys_held.discard(key)
    if key == Key.esc:
        quit_flag.set()
        return False   # stop listener

# --------------------------------------------------------------------------- #
async def control_loop(drone: projectairsim.Drone):
    dt = 1.0 / LOOP_HZ
    print("\nControls: W/S=North/South  A/D=West/East  Q/E=Up/Down"
          "  Space=hover  Esc=land+quit\n")

    while not quit_flag.is_set():
        vn = ve = vd = 0.0

        if KeyCode.from_char('w') in keys_held:  vn += SPEED
        if KeyCode.from_char('s') in keys_held:  vn -= SPEED
        if KeyCode.from_char('d') in keys_held:  ve += SPEED
        if KeyCode.from_char('a') in keys_held:  ve -= SPEED
        if KeyCode.from_char('q') in keys_held:  vd -= SPEED   # up = negative down
        if KeyCode.from_char('e') in keys_held:  vd += SPEED

        await drone.move_by_velocity_async(
            v_north=vn, v_east=ve, v_down=vd, duration=dt * 2
        )
        await asyncio.sleep(dt)


async def main():
    client = projectairsim.ProjectAirSimClient()
    client.connect()

    world = projectairsim.World(
        client, _SCENE_NAME, sim_config_path=_SIM_CONFIG_PATH, delay_after_load_sec=2
    )
    drone = projectairsim.Drone(client, world, "Drone1")

    drone.enable_api_control()
    drone.arm()

    print("Taking off…")
    await await_task(drone.takeoff_async())
    print("Airborne. Use keyboard to fly.")

    # Start global keyboard listener in background thread
    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()

    try:
        await control_loop(drone)
    finally:
        listener.stop()
        print("\nLanding…")
        try:
            await await_task(drone.land_async())
        except Exception:
            pass
        drone.disarm()
        drone.disable_api_control()
        client.disconnect()
        print("Done.")


async def await_task(coro):
    task = await coro
    await task


if __name__ == "__main__":
    asyncio.run(main())
