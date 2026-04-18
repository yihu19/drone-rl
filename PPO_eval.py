import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drone_env import DroneEnv


parser = argparse.ArgumentParser(description="Evaluate custom PPO policy for ProjectAirSim Drone")
parser.add_argument(
    "--ckpt",
    type=str,
    default="./ppo_custom_checkpoints/ppo_actor.pt",
    help="path to trained actor checkpoint",
)
parser.add_argument("--episodes", type=int, default=10, help="number of evaluation episodes")
parser.add_argument("--max-steps", type=int, default=300, help="max steps per episode")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument(
    "--stochastic",
    action="store_true",
    help="sample action from Gaussian policy instead of using deterministic mean action",
)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

STATE_DIM = 9
ACTION_DIM = 3
ACTION_LOW = -5.0
ACTION_HIGH = 5.0


class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mu_head = nn.Linear(128, ACTION_DIM)
        self.sigma_head = nn.Linear(128, ACTION_DIM)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = ACTION_HIGH * torch.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x)) + 1e-4
        sigma = torch.clamp(sigma, min=1e-4, max=2.0)
        return mu, sigma


class PPOEvaluator:
    def __init__(self, ckpt_path):
        self.actor = ActorNet().float()

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.actor.load_state_dict(state_dict)
        self.actor.eval()

    def select_action(self, state, stochastic=False):
        state_t = torch.from_numpy(state).float().unsqueeze(0)

        with torch.no_grad():
            mu, sigma = self.actor(state_t)

            if stochastic:
                dist = torch.distributions.Normal(mu, sigma)
                action = dist.sample()
            else:
                action = mu

            action = torch.clamp(action, ACTION_LOW, ACTION_HIGH)

        return action.squeeze(0).cpu().numpy().astype(np.float32)


def main():
    print("Please make sure the Blocks.sh simulator is running, then press Enter to continue...")
    input()

    env = DroneEnv()
    evaluator = PPOEvaluator(args.ckpt)

    episode_rewards = []
    episode_lengths = []
    success_count = 0

    try:
        for ep in range(args.episodes):
            state, _ = env.reset(seed=args.seed + ep)
            ep_reward = 0.0
            success = False

            for t in range(args.max_steps):
                action = evaluator.select_action(state, stochastic=args.stochastic)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                ep_reward += reward
                state = next_state

                # Same success condition as env: distance < 2.0
                dx, dy, dz = state[6], state[7], state[8]
                dist = float(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
                if dist < 2.0:
                    success = True

                if done:
                    break

            episode_rewards.append(ep_reward)
            episode_lengths.append(t + 1)
            success_count += int(success)

            print(
                f"Episode {ep + 1}/{args.episodes} | "
                f"Reward: {ep_reward:.2f} | "
                f"Steps: {t + 1} | "
                f"Success: {success}"
            )

    finally:
        env.close()

    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    std_reward = float(np.std(episode_rewards)) if episode_rewards else 0.0
    mean_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    success_rate = 100.0 * success_count / max(1, args.episodes)

    print("\n===== Evaluation Summary =====")
    print(f"Checkpoint      : {args.ckpt}")
    print(f"Episodes        : {args.episodes}")
    print(f"Mean Reward     : {mean_reward:.2f}")
    print(f"Std Reward      : {std_reward:.2f}")
    print(f"Mean Length     : {mean_length:.2f}")
    print(f"Success Rate    : {success_rate:.2f}%")
    print(f"Policy Mode     : {'stochastic' if args.stochastic else 'deterministic (mu)'}")


if __name__ == "__main__":
    main()