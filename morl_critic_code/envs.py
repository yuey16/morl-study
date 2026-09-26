"""Six two-objective MuJoCo tasks and IEEE33 with five objectives."""

from pathlib import Path
import sys

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
    "ieee33_pv_rich": "ieee33-pv-rich-5obj-v0",
}
GRID_SCALES = np.array(
    [
        5324.122194342315,
        0.8549682647571899,
        0.16309134631319466,
        0.5152177680326882,
        0.4318290247265395,
    ],
    dtype=np.float64,
)


class Environment(gym.Env):
    """Normalize actions/rewards; count actual transitions and training observations."""

    def __init__(self, name, seed, smoke=False, split="train"):
        self.is_grid = name == "ieee33_pv_rich"
        self.smoke, self.split = smoke, split
        if self.is_grid:
            root = Path(__file__).resolve().parent / "grid"
            sys.path.insert(0, str(root / "src"))
            from residential_grid_morl import ResidentialBESSEVRenewables5ObjEnv

            self.env = ResidentialBESSEVRenewables5ObjEnv(
                root / "configs/v4/ieee33_pv_rich.yaml"
            )
            self.scales = GRID_SCALES
            self.spec = gym.envs.registration.EnvSpec(
                ENVIRONMENTS[name], max_episode_steps=25 if smoke else 96
            )
        else:
            self.env = mo_gym.make(
                ENVIRONMENTS[name], max_episode_steps=25 if smoke else 1000
            )
            self.scales = np.ones(2)
            self.spec = self.env.spec
        self.observation_space = self.env.observation_space
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, self.env.action_space.shape, np.float32
        )
        self.reward_dim = self.env.unwrapped.reward_dim
        self.reward_space = gym.spaces.Box(
            -np.inf, np.inf, (self.reward_dim,), np.float32
        )
        self.scale = (self.env.action_space.high - self.env.action_space.low) / 2
        self.bias = (self.env.action_space.high + self.env.action_space.low) / 2
        self.seed, self.resets, self.steps = seed, 0, 0
        self.before_step = self.observation_observer = None
        self.action_space.seed(seed)

    def reset(self, *, seed=None, options=None):
        seed = self.seed + self.resets if seed is None else seed
        super().reset(seed=seed)
        self.resets += 1
        self.episode_steps = 0
        if self.is_grid and options is None:
            options = {"split": self.split}
        observation, info = self.env.reset(seed=seed, options=options)
        if self.observation_observer is not None:
            self.observation_observer(observation)
        return observation, info

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
        self.episode_steps += 1
        if self.is_grid:
            terminated = bool(
                terminated or truncated or (self.smoke and self.episode_steps >= 25)
            )
            truncated = False
        if self.observation_observer is not None:
            self.observation_observer(observation)
        reward = (np.asarray(reward, np.float64) / self.scales).astype(np.float32)
        return observation, reward, terminated, truncated, info

    def close(self):
        self.env.close()


def preferences(count=11, dimensions=2, seed=891041):
    if dimensions == 2:
        first = np.linspace(0, 1, count, dtype=np.float32)
        return np.column_stack((first, 1 - first))
    if count <= dimensions:
        return np.eye(dimensions, dtype=np.float32)[:count]
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            np.eye(dimensions),
            np.full((1, dimensions), 1 / dimensions),
            rng.dirichlet(np.ones(dimensions), count - dimensions - 1),
        ]
    ).astype(np.float32)
