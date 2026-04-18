"""
PPO training script - ProjectAirSim drone
Using PPO algorithm from stable-baselines3
"""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from drone_env import DroneEnv

print("Please make sure the Blocks.sh simulator is running, then press Enter to continue...")
input()

env = DroneEnv()

model = PPO(
    policy="MlpPolicy",
    env=env,
    device="cpu",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./ppo_drone_logs/",
)

checkpoint_cb = CheckpointCallback(
    save_freq=10000,
    save_path="./ppo_checkpoints/",
    name_prefix="drone_ppo",
)

print("Starting PPO training...")
print("You can watch the drone flying in real time in the simulator window!")
print("Press Ctrl+C to interrupt and save at any time\n")

try:
    model.learn(
        total_timesteps=500_000,
        callback=checkpoint_cb,
        progress_bar=True,
    )
    model.save("drone_ppo_final")
    print("\n✅ Training completed! Model saved as drone_ppo_final.zip")
except KeyboardInterrupt:
    model.save("drone_ppo_interrupted")
    print("\n⚠️ Training interrupted. Model saved as drone_ppo_interrupted.zip")
finally:
    env.close()