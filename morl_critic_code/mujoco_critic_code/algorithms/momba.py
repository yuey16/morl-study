"""Momba with scalar, multi-head, or independent objective critics."""

from pathlib import Path
import sys

import numpy as np
import torch

dependency = Path(__file__).resolve().parents[1] / "_deps" / "momba"
if not (dependency / "momba" / "agent.py").exists():
    raise ImportError("Run `python prepare.py` before using Momba.")
sys.path.insert(0, str(dependency))

from momba.agent import AgentConfig, AgentNets, BinsConfig
from momba.net_builder import SimbaBuilder
from momba.preference_sampling import EpisodicPreferenceSampler
from .momba_objectives import ObjectiveMomba, SplitCritic


class Momba:
    def __init__(self, env, args):
        self.device = torch.device(args.device)
        self.warmup = 16 if args.smoke else 5000
        self.batch_size = 32 if args.smoke else 256
        self.global_step = 0
        self.updates = 0
        self.gamma = 0.99
        self.rng = np.random.default_rng(args.seed)
        self.architecture = args.architecture
        od, ad = env.observation_space.shape[0], env.action_space.shape[0]
        config = AgentConfig(
            obs_dim=od,
            action_dim=ad,
            reward_dim=2,
            action_space=env.action_space,
            observation_space=env.observation_space,
            num_envs=1,
            tau=0.005,
            gamma=0.99,
            best_return_scale=3.0,
            normalize_observations=True,
            normalize_rewards=True,
            alpha=0.2,
            target_entropy=-ad,
            preference_alignment_regularization=False,
            envelope=False,
            envelope_samples=32,
            envelope_concentration=5.0,
        )
        bins = BinsConfig(101, -5.0, 5.0)
        builder = SimbaBuilder(
            config,
            True,
            args.architecture == "multihead",
            bins,
            True,
            False,
            False,
            False,
            True,
            128,
            1,
            args.critic_width or 256,
            2,
        )
        if args.architecture == "split":
            nets = AgentNets(
                builder._make_policy_net(),
                SplitCritic(builder),
                SplitCritic(builder),
                2,
                True,
                True,
            )
        else:
            nets = builder.build()
        self.impl = ObjectiveMomba(
            nets, config, bins, args.architecture + "_" + args.twin_selection
        ).to(self.device)
        self.policy = self.impl.policy
        self.q_nets = [self.impl.qf1, self.impl.qf2]
        self.target_q_nets = [self.impl.qf1_target, self.impl.qf2_target]
        self.policy_optim = torch.optim.Adam(
            self.policy.parameters(), lr=args.actor_lr or 1e-4
        )
        self.q_optim = torch.optim.Adam(
            list(self.q_nets[0].parameters()) + list(self.q_nets[1].parameters()),
            lr=args.critic_lr or 1e-4,
        )
        self.alpha_optim = torch.optim.Adam([self.impl.log_alpha], lr=1e-4)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(
                torch.Generator(device="cpu").manual_seed(args.seed + 1).get_state()
            )
            self.sampler = EpisodicPreferenceSampler(2, self.device)
        capacity = min(args.steps, 1_000_000)
        self.replay = {
            name: np.empty((capacity, size), np.float32)
            for name, size in [
                ("observations", od),
                ("actions", ad),
                ("preferences", 2),
                ("rewards", 2),
                ("next_observations", od),
            ]
        }
        self.replay["dones"] = np.empty(capacity, bool)
        self.size, self.position = 0, 0
        # Keep the post-construction CPU stream independent of critic count.
        torch.set_rng_state(
            torch.Generator(device="cpu").manual_seed(args.seed + 2).get_state()
        )

    @torch.no_grad()
    def act(self, observation, preference, deterministic=False):
        obs = torch.as_tensor(observation, device=self.device, dtype=torch.float32)[
            None
        ]
        weight = torch.as_tensor(preference, device=self.device, dtype=torch.float32)[
            None
        ]
        actions = self.impl.get_action(self.impl.observation_normalizer(obs), weight)
        return actions[2 if deterministic else 0][0].cpu().numpy()

    def eval(self, observation, preference):
        return self.act(observation, preference, deterministic=True)

    def add(
        self,
        observation,
        action,
        preference,
        reward,
        next_observation,
        terminal,
        truncated,
    ):
        values = [observation, action, preference, reward, next_observation, terminal]
        for (name, array), value in zip(self.replay.items(), values):
            array[self.position] = value
        self.position = (self.position + 1) % len(self.replay["dones"])
        self.size = min(self.size + 1, len(self.replay["dones"]))
        with torch.no_grad():
            self.impl.observation_normalizer.update(
                torch.as_tensor(observation, dtype=torch.float32, device=self.device)[
                    None
                ]
            )
            self.impl.reward_normalizer.update(
                torch.as_tensor(reward, dtype=torch.float32, device=self.device)[None],
                torch.tensor([terminal or truncated], device=self.device),
            )

    def update(self):
        indices = self.rng.integers(self.size, size=self.batch_size)
        batch = {
            name: torch.as_tensor(array[indices], device=self.device)
            for name, array in self.replay.items()
        }
        self.impl.observation_normalizer.apply_(batch["observations"])
        self.impl.observation_normalizer.apply_(batch["next_observations"])
        self.impl.reward_normalizer.apply_(batch["rewards"])
        critic_loss = self.impl.critic_loss(
            batch["observations"],
            batch["actions"],
            batch["next_observations"],
            batch["rewards"],
            batch["dones"],
            batch["preferences"],
        )
        self.q_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.q_optim.step()
        actor_loss = self.impl.actor_loss(batch["observations"], batch["preferences"])
        self.policy_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.policy_optim.step()
        alpha_loss, _ = self.impl.alpha_loss(
            batch["observations"], batch["preferences"]
        )
        self.alpha_optim.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optim.step()
        self.impl.normalize_weights()
        self.impl.soft_update_params()
        self.updates += 1
        if not torch.isfinite(torch.stack((critic_loss, actor_loss, alpha_loss))).all():
            raise FloatingPointError("Non-finite Momba training loss.")

    def train(self, env, steps):
        observation, _ = env.reset()
        for step in range(1, steps + 1):
            weight = self.sampler.preference.cpu().numpy().copy()
            action = (
                env.action_space.sample()
                if step <= self.warmup
                else self.act(observation, weight)
            )
            next_observation, reward, terminal, truncated, _ = env.step(action)
            self.add(
                observation,
                action,
                weight,
                reward,
                next_observation,
                terminal,
                truncated,
            )
            self.global_step = step
            if step > self.warmup + 1:
                self.update()
            observation = next_observation
            if terminal or truncated:
                observation, _ = env.reset()
                self.sampler.step(True)
