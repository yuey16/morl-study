"""Six standard, two-objective MuJoCo tasks."""

import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np

ENVIRONMENTS = {
    "ant": "mo-ant-2obj-v5",
    "halfcheetah": "mo-halfcheetah-v5",
    "walker2d": "mo-walker2d-v5",
    "hopper": "mo-hopper-2obj-v5",
    "swimmer": "mo-swimmer-v5",
    "humanoid": "mo-humanoid-v5",
}


class Environment(gym.Env):
    """Normalize the action interface and count actual transitions."""

    def __init__(self, name, seed, smoke=False):
        self.env = mo_gym.make(
            ENVIRONMENTS[name], max_episode_steps=25 if smoke else 1000
        )
        self.observation_space = self.env.observation_space
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, self.env.action_space.shape, np.float32
        )
        self.reward_dim = self.env.unwrapped.reward_dim
        self.reward_space = self.env.unwrapped.reward_space
        self.spec = self.env.spec
        self.scale = (self.env.action_space.high - self.env.action_space.low) / 2
        self.bias = (self.env.action_space.high + self.env.action_space.low) / 2
        self.seed, self.resets, self.steps = seed, 0, 0
        self.before_step = None
        self.action_space.seed(seed)

    def reset(self, *, seed=None, options=None):
        seed = self.seed + self.resets if seed is None else seed
        super().reset(seed=seed)
        self.resets += 1
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        if self.before_step is not None:
            self.before_step(self.steps)
        action = np.asarray(action, dtype=np.float32)
        if not np.isfinite(action).all():
            raise ValueError("Non-finite action.")
        observation, reward, terminated, truncated, info = self.env.step(
            np.clip(action, -1, 1) * self.scale + self.bias
        )
        self.steps += 1
        return observation, np.asarray(reward, np.float32), terminated, truncated, info

    def close(self):
        self.env.close()


def preferences(count=11):
    first = np.linspace(0, 1, count, dtype=np.float32)
    return np.column_stack((first, 1 - first))
