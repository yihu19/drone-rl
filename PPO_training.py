import argparse
import os
import pickle
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from drone_env import DroneEnv


parser = argparse.ArgumentParser(description="Custom PPO for ProjectAirSim Drone")
parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--log-interval", type=int, default=10, help="log interval")
parser.add_argument("--episodes", type=int, default=50, help="number of training episodes")
parser.add_argument("--max-steps", type=int, default=300, help="max steps per episode")
parser.add_argument("--actor-lr", type=float, default=1e-4, help="actor learning rate")
parser.add_argument("--critic-lr", type=float, default=3e-4, help="critic learning rate")
parser.add_argument("--save-dir", type=str, default="./ppo_custom_checkpoints", help="checkpoint directory")
parser.add_argument("--plot-path", type=str, default="./ppo_custom_reward.png", help="reward plot path")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

TrainingRecord = namedtuple("TrainingRecord", ["ep", "reward"])
Transition = namedtuple("Transition", ["s", "a", "a_log_p", "r", "s_", "done"])


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

        # Output mean in action range [-5, 5]
        mu = ACTION_HIGH * torch.tanh(self.mu_head(x))

        # Positive std; clamp for stability
        sigma = F.softplus(self.sigma_head(x)) + 1e-4
        sigma = torch.clamp(sigma, min=1e-4, max=2.0)
        return mu, sigma


class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, 128)
        self.fc2 = nn.Linear(128, 128)
        self.v_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.v_head(x)


class Agent:
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    buffer_capacity = 2048
    batch_size = 64

    def __init__(self):
        self.training_step = 0
        self.anet = ActorNet().float()
        self.cnet = CriticNet().float()
        self.buffer = []

        self.optimizer_a = optim.Adam(self.anet.parameters(), lr=args.actor_lr)
        self.optimizer_c = optim.Adam(self.cnet.parameters(), lr=args.critic_lr)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            mu, sigma = self.anet(state)

        dist = Normal(mu, sigma)
        action = dist.sample()
        action = torch.clamp(action, ACTION_LOW, ACTION_HIGH)

        # Sum log-prob over action dimensions
        action_log_prob = dist.log_prob(action).sum(dim=1)

        return action.squeeze(0).cpu().numpy(), action_log_prob.item()

    def get_value(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            state_value = self.cnet(state)
        return state_value.item()

    def save_param(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.anet.state_dict(), os.path.join(save_dir, "ppo_actor.pt"))
        torch.save(self.cnet.state_dict(), os.path.join(save_dir, "ppo_critic.pt"))

    def store(self, transition):
        self.buffer.append(transition)
        return len(self.buffer) >= self.buffer_capacity

    def update(self):
        self.training_step += 1

        s = torch.tensor(np.array([t.s for t in self.buffer]), dtype=torch.float32)
        a = torch.tensor(np.array([t.a for t in self.buffer]), dtype=torch.float32)
        r = torch.tensor(np.array([t.r for t in self.buffer]), dtype=torch.float32).view(-1, 1)
        s_ = torch.tensor(np.array([t.s_ for t in self.buffer]), dtype=torch.float32)
        done = torch.tensor(np.array([t.done for t in self.buffer]), dtype=torch.float32).view(-1, 1)

        old_action_log_probs = torch.tensor(
            np.array([t.a_log_p for t in self.buffer]), dtype=torch.float32
        ).view(-1, 1)

        # Reward normalization, similar spirit to PPO_pendulum.py
        r = (r - r.mean()) / (r.std() + 1e-5)

        with torch.no_grad():
            target_v = r + args.gamma * self.cnet(s_) * (1.0 - done)

        adv = (target_v - self.cnet(s)).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        total_size = len(self.buffer)

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(
                SubsetRandomSampler(range(total_size)),
                self.batch_size,
                drop_last=False,
            ):
                mu, sigma = self.anet(s[index])
                dist = Normal(mu, sigma)

                action_log_probs = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(action_log_probs - old_action_log_probs[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_param,
                    1.0 + self.clip_param,
                ) * adv[index]

                action_loss = -torch.min(surr1, surr2).mean()

                self.optimizer_a.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.anet.parameters(), self.max_grad_norm)
                self.optimizer_a.step()

                value_loss = F.smooth_l1_loss(self.cnet(s[index]), target_v[index])

                self.optimizer_c.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.cnet.parameters(), self.max_grad_norm)
                self.optimizer_c.step()

        self.buffer.clear()


def main():
    print("Please make sure the Blocks.sh simulator is running, then press Enter to continue...")
    input()

    env = DroneEnv()
    agent = Agent()

    os.makedirs(args.save_dir, exist_ok=True)

    training_records = []
    running_reward = None
    best_running_reward = -float("inf")

    try:
        for i_ep in range(args.episodes):
            state, _ = env.reset(seed=args.seed + i_ep)
            score = 0.0

            for t in range(args.max_steps):
                action, action_log_prob = agent.select_action(state)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                if agent.store(
                    Transition(state, action, action_log_prob, reward, next_state, done)
                ):
                    agent.update()

                score += reward
                state = next_state

                if done:
                    break

            # update remaining samples if buffer not empty for too long
            if len(agent.buffer) > 0 and len(agent.buffer) >= agent.batch_size:
                agent.update()

            if running_reward is None:
                running_reward = score
            else:
                running_reward = running_reward * 0.9 + score * 0.1

            training_records.append(TrainingRecord(i_ep, running_reward))

            if i_ep % args.log_interval == 0:
                print(
                    f"Ep {i_ep}\t"
                    f"Episode reward: {score:.2f}\t"
                    f"Moving average reward: {running_reward:.2f}"
                )

            if running_reward > best_running_reward:
                best_running_reward = running_reward
                agent.save_param(args.save_dir)
                with open(os.path.join(args.save_dir, "ppo_training_records.pkl"), "wb") as f:
                    pickle.dump(training_records, f)

        # final save
        agent.save_param(args.save_dir)
        with open(os.path.join(args.save_dir, "ppo_training_records.pkl"), "wb") as f:
            pickle.dump(training_records, f)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")
        agent.save_param(args.save_dir)
        with open(os.path.join(args.save_dir, "ppo_training_records.pkl"), "wb") as f:
            pickle.dump(training_records, f)

    finally:
        env.close()

    plt.plot([r.ep for r in training_records], [r.reward for r in training_records])
    plt.title("Custom PPO - Drone")
    plt.xlabel("Episode")
    plt.ylabel("Moving averaged episode reward")
    plt.grid(True)
    plt.savefig(args.plot_path)
    plt.show()


if __name__ == "__main__":
    main()