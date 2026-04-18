"""
ProjectAirSim Drone Gym Environment
"""

import asyncio
import math
import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from projectairsim import ProjectAirSimClient, World, Drone
from projectairsim.types import Pose, Vector3, Quaternion

_SIM_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "client", "python", "example_user_scripts", "sim_config",
)
_SCENE_NAME = "scene_basic_drone.jsonc"

# Spawn pose from scene_basic_drone.jsonc: xyz="-1.0 8.0 -4.0", rpy-deg="0 0 -45"
_yaw = math.radians(-45)
_SPAWN_POSE = Pose({
    "translation": Vector3({"x": -1.0, "y": 8.0, "z": -4.0}),
    "rotation": Quaternion({"w": math.cos(_yaw / 2), "x": 0.0, "y": 0.0, "z": math.sin(_yaw / 2)}),
})


class DroneEnv(gym.Env):
    """
    ProjectAirSim drone Gym environment.
    Goal: fly from spawn point to target (9, 8, -9) in NED coordinates.
    NED: North=X+, East=Y+, Down=Z+ (altitude is negative Z).
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Box(
            low=np.float32(-5.0),
            high=np.float32(5.0),
            shape=(3,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32,
        )

        self.target = np.array([9.0, 8.0, -9.0], dtype=np.float32)
        self.max_steps = 300
        self.step_count = 0

        self.client = None
        self.world = None
        self.drone = None
        self._episode = 0

        self.loop = asyncio.new_event_loop()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def _connect(self):
        """Connect once per training run and load the scene (fresh Blocks.sh required)."""
        self.client = ProjectAirSimClient()
        self.client.connect()
        print("  [OK] connected to sim")

        # Load scene exactly once — works on a fresh Blocks.sh that has no scene yet.
        # Do NOT call this again on subsequent resets; use set_pose instead.
        self.world = World(self.client, _SCENE_NAME,
                           sim_config_path=_SIM_CONFIG_PATH,
                           delay_after_load_sec=2)
        self.drone = Drone(self.client, self.world, "Drone1")
        print("  [OK] Drone initialized")

    def _get_obs(self):
        state = self.drone.get_ground_truth_kinematics()
        pos = state["pose"]["position"]
        vel = state["twist"]["linear"]

        x, y, z   = float(pos["x"]), float(pos["y"]), float(pos["z"])
        vx, vy, vz = float(vel["x"]), float(vel["y"]), float(vel["z"])

        dx = self.target[0] - x
        dy = self.target[1] - y
        dz = self.target[2] - z

        return np.array([x, y, z, vx, vy, vz, dx, dy, dz], dtype=np.float32)

    def _compute_reward(self, obs):
        dx, dy, dz = obs[6], obs[7], obs[8]
        dist = float(np.sqrt(dx**2 + dy**2 + dz**2))
        reward = -dist * 0.1
        if dist < 2.0:
            reward += 100.0
        x, y, z = float(obs[0]), float(obs[1]), float(obs[2])
        if abs(x) > 50 or abs(y) > 50 or abs(z) > 30:
            reward -= 50.0
        return float(reward)

    def _is_done(self, obs):
        dx, dy, dz = obs[6], obs[7], obs[8]
        dist = float(np.sqrt(dx**2 + dy**2 + dz**2))
        if dist < 2.0:
            print(f"  reached target dist={dist:.2f}m")
            return True
        x, y, z = float(obs[0]), float(obs[1]), float(obs[2])
        if abs(x) > 50 or abs(y) > 50 or abs(z) > 30:
            print(f"  out of bounds pos=({x:.1f},{y:.1f},{z:.1f})")
            return True
        return self.step_count >= self.max_steps

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._episode += 1
        print(f"\n--- Episode {self._episode} ---")

        if self.client is None:
            print("connecting to sim and loading scene...")
            self._connect()

        # Teleport drone back to spawn without reloading the scene.
        # Reloading would trigger UE garbage collection which deadlocks the render thread.
        try:
            self.drone.disarm()
        except Exception:
            pass
        self.drone.set_pose(_SPAWN_POSE, True)   # True = reset_kinematics (zero velocity)

        self.drone.enable_api_control()
        self.drone.arm()

        takeoff_task = self._run(self.drone.takeoff_async())
        self._run(takeoff_task)
        print("  [OK] takeoff done")

        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        move_task = self._run(
            self.drone.move_by_velocity_async(
                v_north=float(action[0]),
                v_east=float(action[1]),
                v_down=float(action[2]),
                duration=0.1,
            )
        )
        self._run(move_task)

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        done = self._is_done(obs)
        return obs, reward, done, False, {}

    def close(self):
        if self.drone:
            try:
                self.drone.disarm()
                self.drone.disable_api_control()
            except Exception:
                pass
        if self.client:
            try:
                self.client.disconnect()
            except Exception:
                pass
        if not self.loop.is_closed():
            self.loop.close()
