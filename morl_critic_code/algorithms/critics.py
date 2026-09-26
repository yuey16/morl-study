"""Change critic sharing while retaining the actor and the original losses."""

from copy import deepcopy
from itertools import chain

import torch
from torch import nn


class BranchedCritic(nn.Module):
    def __init__(self, reference, architecture, container="net"):
        super().__init__()
        layers = list(getattr(reference, container).children())
        linear = [i for i, layer in enumerate(layers) if isinstance(layer, nn.Linear)]
        cut = linear[1] if architecture == "multihead" else 0
        self.trunk = deepcopy(nn.Sequential(*layers[:cut]))
        self.heads = nn.ModuleList()
        for i in range(layers[-1].out_features):
            output = deepcopy(layers[-1])
            output.weight = nn.Parameter(output.weight[i : i + 1].detach().clone())
            output.bias = nn.Parameter(output.bias[i : i + 1].detach().clone())
            output.out_features = 1
            self.heads.append(nn.Sequential(*deepcopy(layers[cut:-1]), output))

    def forward(self, *inputs):
        hidden = self.trunk(torch.cat(inputs, dim=-1))
        return torch.cat([head(hidden) for head in self.heads], dim=-1)


class SplitMPOCritic(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                nn.Sequential(deepcopy(reference.first), deepcopy(head))
                for head in reference.objectives
            ]
        )

    def forward(self, observation, latent_action):
        inputs = torch.cat((observation, latent_action.tanh()), dim=-1)
        return torch.cat([head(inputs) for head in self.heads], dim=-1)


class SharedMPOCritic(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.first = deepcopy(reference.first)
        self.hidden = deepcopy(reference.objectives[0][:-1])
        last = reference.objectives[0][-1]
        self.output = nn.Linear(last.in_features, len(reference.objectives)).to(
            last.weight
        )
        with torch.no_grad():
            for i, head in enumerate(reference.objectives):
                self.output.weight[i].copy_(head[-1].weight[0])
                self.output.bias[i].copy_(head[-1].bias[0])

    def forward(self, observation, latent_action):
        inputs = torch.cat((observation, latent_action.tanh()), dim=-1)
        return self.output(self.hidden(self.first(inputs)))


def optimizer_like(original, parameters):
    return type(original)(parameters, **original.defaults)


def install(agent, method, architecture, env, width=None):
    owners = (
        [p.wrapped for p in agent.population]
        if hasattr(agent, "population")
        else [agent]
    )
    device = torch.device(agent.device)
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        for owner in owners:
            if method == "mo_mpo":
                reference = owner.critic
                if width is not None and width != 400:
                    from .mo_mpo import ObjectiveCritics

                    reference = ObjectiveCritics(
                        env.observation_space.shape[0],
                        env.action_space.shape[0],
                        env.reward_dim,
                        width,
                    ).to(device)
                owner.critic = reference
                if architecture == "split":
                    owner.critic = SplitMPOCritic(reference)
                elif architecture == "shared":
                    owner.critic = SharedMPOCritic(reference)
                owner.critic.to(device)
                owner.target_critic = deepcopy(owner.critic).requires_grad_(False)
                owner.critic_optimizer = optimizer_like(
                    owner.critic_optimizer, owner.critic.parameters()
                )
                continue

            mosac = method == "morld_mosac"
            container = "critic" if mosac else "net"
            references = [owner.qf1, owner.qf2] if mosac else owner.q_nets
            if width is not None and width != 256:
                if mosac:
                    from morl_baselines.single_policy.ser.mosac_continuous_action import (
                        MOSoftQNetwork,
                    )

                    references = [
                        MOSoftQNetwork(
                            env.observation_space.shape,
                            env.action_space.shape,
                            env.reward_dim,
                            [width, width],
                        ).to(device)
                        for _ in references
                    ]
                else:
                    if method == "capql":
                        from morl_baselines.multi_policy.capql.capql import QNetwork
                    else:
                        from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import (
                            QNetwork,
                        )
                    references = [
                        QNetwork(
                            env.observation_space.shape[0],
                            env.action_space.shape[0],
                            env.reward_dim,
                            [width, width],
                        ).to(device)
                        for _ in references
                    ]
            online = (
                references
                if architecture == "shared"
                else [
                    BranchedCritic(q, architecture, container).to(device)
                    for q in references
                ]
            )
            targets = [deepcopy(q).requires_grad_(False) for q in online]
            parameters = chain(*(q.parameters() for q in online))
            if mosac:
                owner.qf1, owner.qf2 = online
                owner.qf1_target, owner.qf2_target = targets
                owner.q_optimizer = optimizer_like(owner.q_optimizer, parameters)
            else:
                owner.q_nets, owner.target_q_nets = online, targets
                owner.q_optim = optimizer_like(owner.q_optim, parameters)
