"""MO-MPO with TD(0) critics and per-objective Gaussian policy improvement."""

from copy import deepcopy
from types import SimpleNamespace
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def normal_log_prob(a, mean, std):
    return (
        -0.5 * ((a - mean) / std).square() - std.log() - 0.5 * math.log(2 * math.pi)
    ).sum(-1)


def objective_e_step(q_values, temperature, epsilon):
    """Inputs [states, sampled actions, objectives]; statewise normalization."""
    q = q_values.detach()
    eta = temperature.clamp_min(1e-08)
    offset = q.max(dim=1, keepdim=True).values
    logits = (q - offset) / eta
    probabilities = logits.softmax(dim=1).detach()
    dual_per_objective = eta * (
        epsilon + (torch.logsumexp(logits, dim=1) - math.log(q.shape[1])).mean(dim=0)
    ) + offset.mean(dim=(0, 1))
    return (probabilities, dual_per_objective.sum())


def gaussian_projection(
    actions,
    objective_probabilities,
    old_mean,
    old_std,
    new_mean,
    new_std,
    multipliers,
    bounds,
):
    """Negative primal and dual losses; disjoint actor and dual gradients.

    The means/covariances in the other partial Gaussian are frozen as in
    App.C.3 equations18-20. KL is old||new, summed over action dimensions.
    """
    (old_mean, old_std) = (old_mean.detach(), old_std.detach())
    weights = objective_probabilities.detach().sum(dim=-1)
    lp_mean = normal_log_prob(actions, new_mean[:, None, :], old_std[:, None, :])
    lp_std = normal_log_prob(actions, old_mean[:, None, :], new_std[:, None, :])
    fit = -(weights * (lp_mean + lp_std)).sum(1).mean()
    kl_mean = (0.5 * ((new_mean - old_mean) / old_std).square()).sum(-1).mean()
    kl_covariance = (
        (new_std.log() - old_std.log() + 0.5 * (old_std / new_std).square() - 0.5)
        .sum(-1)
        .mean()
    )
    kl = torch.stack((kl_mean, kl_covariance))
    primal = fit + (multipliers.detach() * kl).sum()
    dual = (multipliers * (bounds - kl.detach())).sum()
    return (primal, dual, kl)


def td0_targets(rewards, terminal, next_values, gamma):
    return rewards + gamma * (1 - terminal[..., None]) * next_values


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.first = nn.Sequential(
            nn.Linear(obs_dim, 300), nn.LayerNorm(300), nn.Tanh()
        )
        self.hidden = nn.Sequential(nn.Linear(300, 200), nn.ELU())
        self.mean = nn.Linear(200, action_dim)
        self.scale = nn.Linear(200, action_dim)

    def forward(self, obs):
        hidden = self.hidden(self.first(obs))
        return (self.mean(hidden).tanh(), F.softplus(self.scale(hidden)) + 1e-06)


class ObjectiveCritics(nn.Module):
    def __init__(self, obs_dim, action_dim, reward_dim=2, width=400):
        h, last = width, round(0.75 * width)
        super().__init__()
        self.first = nn.Sequential(
            nn.Linear(obs_dim + action_dim, h), nn.LayerNorm(h), nn.Tanh()
        )
        self.objectives = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(h, h),
                    nn.ELU(),
                    nn.Linear(h, last),
                    nn.ELU(),
                    nn.Linear(last, 1),
                )
                for _ in range(reward_dim)
            ]
        )

    def forward(self, obs, latent_action):
        hidden = self.first(torch.cat((obs, latent_action.tanh()), dim=-1))
        return torch.cat([head(hidden) for head in self.objectives], dim=-1)


class Duals(nn.Module):
    def __init__(self, epsilon):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(len(epsilon)))
        self.multipliers = nn.Parameter(torch.ones(2))
        self.register_buffer("epsilon", torch.tensor(epsilon, dtype=torch.float32))
        self.register_buffer("bounds", torch.tensor([0.001, 1e-05]))

    def project(self):
        with torch.no_grad():
            self.temperature.clamp_(min=1e-08)
            self.multipliers.clamp_(min=1e-08)


class FixedPreferencePolicy:
    def __init__(self, env, epsilon, seed, smoke, device, capacity, width):
        self.device = device
        self.epsilon = epsilon
        self.gamma = 0.99
        self.batch_size = 32 if smoke else 512
        self.learning_starts = 8 if smoke else 1000
        self.target_period = 20 if smoke else 200
        self.action_samples = 20
        self.updates = 0
        self.steps = 0
        self.rng = np.random.default_rng(seed)
        (od, ad) = (env.observation_space.shape[0], env.action_space.shape[0])
        self.actor = GaussianActor(od, ad).to(device)
        self.critic = ObjectiveCritics(od, ad, len(epsilon), width).to(device)
        self.target_actor = deepcopy(self.actor).requires_grad_(False)
        self.target_critic = deepcopy(self.critic).requires_grad_(False)
        self.duals = Duals(epsilon).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=0.0003, eps=0.001
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=0.0003, eps=0.001
        )
        self.temperature_optimizer = torch.optim.Adam(
            [self.duals.temperature], lr=0.0003, eps=0.001
        )
        self.kl_optimizer = torch.optim.Adam(
            [self.duals.multipliers], lr=0.0003, eps=0.001
        )
        self.action_scale = (env.action_space.high - env.action_space.low) / 2
        self.action_bias = (env.action_space.high + env.action_space.low) / 2
        self.replay = dict(
            obs=np.empty((capacity, od), np.float32),
            action=np.empty((capacity, ad), np.float32),
            reward=np.empty((capacity, len(epsilon)), np.float32),
            next_obs=np.empty((capacity, od), np.float32),
            terminal=np.empty(capacity, np.float32),
        )

    def physical_action(self, raw):
        return (np.clip(raw, -1, 1) * self.action_scale + self.action_bias).astype(
            np.float32
        )

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        (mean, std) = self.actor(
            torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        )
        raw = mean if deterministic else mean + std * torch.randn_like(mean)
        latent = raw.cpu().numpy()
        return (self.physical_action(latent), latent)

    def eval(self, obs, w=None):
        return self.act(obs, deterministic=True)[0]

    def append(self, obs, latent, reward, next_obs, terminal):
        assert self.steps < len(self.replay["terminal"])
        for key, value in zip(self.replay, [obs, latent, reward, next_obs, terminal]):
            self.replay[key][self.steps] = value
        self.steps += 1

    def update(self):
        ids = self.rng.integers(0, self.steps, size=self.batch_size)
        batch = {
            k: torch.as_tensor(x[ids], device=self.device)
            for (k, x) in self.replay.items()
        }
        (obs, nxt) = (batch["obs"], batch["next_obs"])
        with torch.no_grad():
            (old_mean, old_std) = self.target_actor(obs)
            actions = old_mean[:, None, :] + old_std[:, None, :] * torch.randn(
                len(obs), self.action_samples, old_mean.shape[-1], device=self.device
            )
            replicated = obs[:, None, :].expand(-1, self.action_samples, -1)
            q_values = self.target_critic(replicated, actions)
            (next_mean, next_std) = self.target_actor(nxt)
            next_actions = next_mean[:, None, :] + next_std[
                :, None, :
            ] * torch.randn_like(actions)
            next_value = self.target_critic(
                nxt[:, None, :].expand_as(replicated), next_actions
            ).mean(1)
            targets = td0_targets(
                batch["reward"], batch["terminal"], next_value, self.gamma
            )
        (_, temperature_loss) = objective_e_step(
            q_values, self.duals.temperature, self.duals.epsilon
        )
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        self.temperature_optimizer.step()
        self.duals.project()
        (probabilities, _) = objective_e_step(
            q_values, self.duals.temperature, self.duals.epsilon
        )
        (new_mean, new_std) = self.actor(obs)
        (primal, dual, kl) = gaussian_projection(
            actions,
            probabilities,
            old_mean,
            old_std,
            new_mean,
            new_std,
            self.duals.multipliers,
            self.duals.bounds,
        )
        self.actor_optimizer.zero_grad(set_to_none=True)
        primal.backward()
        self.actor_optimizer.step()
        self.kl_optimizer.zero_grad(set_to_none=True)
        dual.backward()
        self.kl_optimizer.step()
        self.duals.project()
        prediction = self.critic(obs, batch["action"])
        critic_loss = (prediction - targets).square().mean(0).sum()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.updates += 1
        if self.updates % self.target_period == 0:
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic.load_state_dict(self.critic.state_dict())


class MOMPO:
    def __init__(self, env, seed, steps=200000, smoke=False, device="cpu", width=400):
        (self.device, self.gamma, self.global_step) = (device, 0.99, 0)
        self.settings = [
            [0.1, 0.0001],
            [0.1, 0.01],
            [0.1, 0.1],
            [0.01, 0.1],
            [0.0001, 0.1],
        ]
        self.reward_dim = env.unwrapped.reward_dim
        if self.reward_dim == 5:
            self.settings = [
                [0.1 if i == j else 0.0001 for j in range(5)] for i in range(5)
            ] + [[0.1] * 5, [0.01] * 5]
        count = len(self.settings)
        self.population = [
            SimpleNamespace(
                wrapped=FixedPreferencePolicy(
                    env,
                    eps,
                    seed + i,
                    smoke,
                    device,
                    steps // count + (i < steps % count),
                    width,
                ),
                epsilon=eps,
            )
            for (i, eps) in enumerate(self.settings)
        ]

    def train(self, env, total_steps):
        count = len(self.population)
        for i, member in enumerate(self.population):
            per_policy = total_steps // count + (i < total_steps % count)
            p = member.wrapped
            (obs, _) = env.reset()
            for _ in range(per_policy):
                (physical, latent) = p.act(obs)
                (nxt, reward, term, trunc, _) = env.step(physical)
                p.append(obs, latent, reward, nxt, term)
                self.global_step += 1
                if p.steps >= p.learning_starts:
                    p.update()
                obs = nxt
                if term or trunc:
                    (obs, _) = env.reset()
            assert p.steps == per_policy
