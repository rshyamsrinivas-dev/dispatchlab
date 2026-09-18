"""Masked dueling Double DQN with experience replay and a Polyak target network."""

from pathlib import Path
import numpy as np
import torch
from torch import nn
from .fleet import OBS_SIZE, encode


class DuelingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(OBS_SIZE, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, 64)

    def forward(self, x):
        h = self.trunk(x)
        adv = self.advantage(h)
        return self.value(h) + adv - adv.mean(dim=-1, keepdim=True)


class ReplayBuffer:
    def __init__(self, size=30000):
        self.size = size
        self.cursor = 0
        self.count = 0
        self.states = np.empty((size, OBS_SIZE), np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(size, np.int64)
        self.rewards = np.empty(size, np.float32)
        self.done = np.empty(size, np.float32)
        self.masks = np.empty((size, 64), bool)

    def add(self, state, action, reward, nxt, done, mask):
        i = self.cursor
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = nxt
        self.done[i] = done
        self.masks[i] = mask
        self.cursor = (i + 1) % self.size
        self.count = min(self.count + 1, self.size)


class DoubleDQN:
    def __init__(self, seed=0, training=True):
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.online = DuelingNetwork()
        self.target = DuelingNetwork()
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.gamma = 0.97
        self.tau = 0.01
        self.optimizer = (
            torch.optim.Adam(self.online.parameters(), lr=3e-4) if training else None
        )
        self.buffer = ReplayBuffer() if training else None
        self.updates = 0

    @torch.no_grad()
    def q_values(self, obs):
        return self.online(torch.from_numpy(encode(obs)).unsqueeze(0))[0].numpy()

    def act(self, obs, mask):
        return int(np.argmax(np.where(mask, self.q_values(obs), -np.inf)))

    def optimise(self, batch=64):
        b = self.buffer
        if b.count < 1024:
            return None
        idx = self.rng.integers(0, b.count, size=batch)
        state = torch.from_numpy(b.states[idx])
        nxt = torch.from_numpy(b.next_states[idx])
        action = torch.from_numpy(b.actions[idx])
        reward = torch.from_numpy(b.rewards[idx])
        done = torch.from_numpy(b.done[idx])
        mask = torch.from_numpy(b.masks[idx])
        q = self.online(state).gather(1, action[:, None]).squeeze(1)
        with torch.no_grad():
            # Online selects; target evaluates. Mask prevents bootstrapping from
            # physically impossible next actions. Terminal bootstrap is zero.
            next_action = self.online(nxt).masked_fill(~mask, -torch.inf).argmax(1)
            next_q = self.target(nxt).gather(1, next_action[:, None]).squeeze(1)
            target = reward + self.gamma * (1 - done) * next_q
        loss = nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10)
        self.optimizer.step()
        with torch.no_grad():
            for t, p in zip(self.target.parameters(), self.online.parameters()):
                t.lerp_(p, self.tau)
        self.updates += 1
        return float(loss.item())

    def save(self, path):
        # Pure state_dict; no arbitrary Python objects in the checkpoint.
        torch.save(self.online.state_dict(), Path(path))

    @classmethod
    def load(cls, path):
        agent = cls(training=False)
        agent.online.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        agent.online.eval()
        return agent
