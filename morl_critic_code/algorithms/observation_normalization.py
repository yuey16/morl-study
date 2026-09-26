"""Training-only normalization for all new IEEE33 architecture variants.

No new trainable parameters or analytic reward information. Raw replay retained.
"""

import torch
from torch import nn


class ObservationMoments(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.epsilon = 1e-8
        self.clip = 10.0
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("variance", torch.ones(shape, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float64))
        self.register_buffer("observations_seen", torch.tensor(0, dtype=torch.int64))

    @torch.no_grad()
    def observe(self, obs):
        x = torch.as_tensor(obs, dtype=self.mean.dtype, device=self.mean.device)
        assert x.shape == self.mean.shape
        delta = x - self.mean
        total = self.count + 1
        self.variance.copy_(
            (self.variance * self.count + delta.square() * self.count / total) / total
        )
        self.mean.add_(delta / total)
        self.count.copy_(total)
        self.observations_seen.add_(1)

    def forward(self, obs):
        return (
            (obs - self.mean.to(obs.dtype))
            / torch.sqrt(self.variance.to(obs.dtype) + self.epsilon)
        ).clamp(-self.clip, self.clip)


class NormalizedModule(nn.Module):
    def __init__(self, module, normalizer):
        super().__init__()
        self.module = module
        self.normalizer = normalizer

    def forward(self, obs, *args, **kwargs):
        return self.module(self.normalizer(obs), *args, **kwargs)

    def sample(self, obs, *args, **kwargs):
        return self.module.sample(self.normalizer(obs), *args, **kwargs)

    def get_action(self, obs, *args, **kwargs):
        return self.module.get_action(self.normalizer(obs), *args, **kwargs)


def install(agent, method, training):
    normalizer = ObservationMoments(training.observation_space.shape).to(agent.device)
    before = {
        id(p)
        for group in optimizer_list(agent, method)
        for params in group.param_groups
        for p in params["params"]
    }
    if method in ["capql", "gpi_ls"]:
        agent.policy = NormalizedModule(agent.policy, normalizer)
        if hasattr(agent, "target_policy"):
            agent.target_policy = NormalizedModule(agent.target_policy, normalizer)
        agent.q_nets = [NormalizedModule(q, normalizer) for q in agent.q_nets]
        agent.target_q_nets = [
            NormalizedModule(q, normalizer) for q in agent.target_q_nets
        ]
    else:
        for member in agent.population:
            owner = member.wrapped
            names = (
                ["actor", "qf1", "qf2", "qf1_target", "qf2_target"]
                if method == "morld_mosac"
                else ["actor", "target_actor", "critic", "target_critic"]
            )
            for name in names:
                setattr(owner, name, NormalizedModule(getattr(owner, name), normalizer))
    agent.observation_normalizer = normalizer
    after = {
        id(p)
        for group in optimizer_list(agent, method)
        for params in group.param_groups
        for p in params["params"]
    }
    assert before == after
    training.observation_observer = normalizer.observe
    return normalizer


def optimizer_list(agent, method):
    if method in ["capql", "gpi_ls"]:
        return [agent.policy_optim, agent.q_optim]
    return [
        getattr(p.wrapped, name)
        for p in agent.population
        for name in (
            ["actor_optimizer", "q_optimizer"]
            if method == "morld_mosac"
            else ["actor_optimizer", "critic_optimizer"]
        )
    ]
