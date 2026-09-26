"""Minimal interfaces to the existing algorithm implementations."""

from types import SimpleNamespace

import numpy as np
import torch

from .critics import install
from envs import preferences


class FrozenActor:
    """Store an archive policy without copying its replay buffer or environment."""

    def __init__(self, actor, device):
        self.actor, self.device = actor, device

    @torch.no_grad()
    def eval(self, observation, weight=None):
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)[
            None
        ]
        return self.actor.get_action(obs)[0][0].cpu().numpy()


def archive():
    from morl_baselines.common.pareto import ParetoArchive

    class ActorArchive(ParetoArchive):
        def add(self, candidate, evaluation):
            snapshot = SimpleNamespace(
                id=candidate.id,
                weights=candidate.weights.copy(),
                wrapped=FrozenActor(candidate.wrapped.actor, candidate.wrapped.device),
            )
            super().add(snapshot, evaluation.copy())
            for member in self.individuals:
                member.wrapped.actor.requires_grad_(False)

    return ActorArchive()


def build(env, args):
    if args.method == "momba":
        from .momba import Momba

        return Momba(env, args)
    if args.method == "capql":
        from morl_baselines.multi_policy.capql.capql import CAPQL

        agent = CAPQL(
            env,
            seed=args.seed,
            device=args.device,
            log=False,
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,
            num_q_nets=2,
            net_arch=[256, 256],
            batch_size=32 if args.smoke else 256,
            buffer_size=min(args.steps, 1_000_000),
            alpha=0.2,
            learning_starts=32 if args.smoke else 1000,
            gradient_updates=1,
        )
    elif args.method == "morld_mosac":
        from morl_baselines.multi_policy.morld.morld import MORLD

        agent = MORLD(
            env,
            seed=args.seed,
            device=args.device,
            log=False,
            gamma=0.99,
            policy_name="MOSAC",
            pop_size=5,
            exchange_every=20 if args.smoke else 10000,
            shared_buffer=True,
            sharing_mechanism=[],
            update_passes=10,
            weight_adaptation_method="PSA",
            policy_args=dict(
                device=args.device,
                buffer_size=min(args.steps, 200000),
                batch_size=32 if args.smoke else 128,
                learning_starts=16 if args.smoke else 1000,
            ),
        )
        agent.archive = archive()
    elif args.method == "gpi_ls":
        from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import (
            GPILSContinuousAction,
        )

        agent = GPILSContinuousAction(
            env,
            seed=args.seed,
            device=args.device,
            log=False,
            gradient_updates=1 if args.smoke else 20,
            batch_size=32 if args.smoke else 128,
            learning_starts=16 if args.smoke else 100,
            buffer_size=min(args.steps, 200000),
        )
    else:
        from .mo_mpo import MOMPO

        agent = MOMPO(
            env,
            args.seed,
            args.steps,
            args.smoke,
            args.device,
            400,
        )
    install(agent, args.method, args.architecture, env, args.critic_width)
    owners = (
        [p.wrapped for p in agent.population]
        if hasattr(agent, "population")
        else [agent]
    )
    for owner in owners:
        actor = getattr(owner, "policy_optim", getattr(owner, "actor_optimizer", None))
        critic = getattr(
            owner,
            "q_optim",
            getattr(owner, "q_optimizer", getattr(owner, "critic_optimizer", None)),
        )
        for optimizer, value in [(actor, args.actor_lr), (critic, args.critic_lr)]:
            if value is not None:
                for group in optimizer.param_groups:
                    group["lr"] = value
    return agent


def train(agent, env, selection, args):
    if args.method in ("momba", "mo_mpo"):
        agent.train(env, args.steps)
        return
    options = dict(
        total_timesteps=args.steps,
        eval_env=selection,
        ref_point=np.array([-10000.0, -10000.0]),
        checkpoints=False,
    )
    if args.method == "gpi_ls":
        options.update(
            timesteps_per_iter=20 if args.smoke else 10000,
            num_eval_episodes_for_front=1 if args.smoke else 5,
            weight_selection_algo="gpi-ls",
        )
    elif args.method == "morld_mosac":
        options.update(num_eval_episodes_for_front=1 if args.smoke else 5)
    agent.train(**options)


def candidates(agent, method, count=11):
    if method == "mo_mpo":
        return list(range(len(agent.population)))
    if method == "morld_mosac":
        estimates = np.asarray(agent.archive.evaluations)
        if len(estimates) <= count:
            return list(range(len(estimates)))
        indices = list(
            dict.fromkeys(np.argmax(preferences(count) @ estimates.T, axis=1).tolist())
        )
        normalized = (estimates - estimates.min(0)) / np.maximum(
            np.ptp(estimates, axis=0), 1e-8
        )
        while len(indices) < count:
            distances = (
                ((normalized[:, None] - normalized[indices][None]) ** 2).sum(-1).min(1)
            )
            distances[indices] = -1
            indices.append(int(np.argmax(distances)))
        return indices
    return list(preferences(count))


@torch.no_grad()
def action(agent, method, observation, candidate):
    if method == "mo_mpo":
        return agent.population[candidate].wrapped.eval(observation)
    if method == "morld_mosac":
        owner = agent.archive.individuals[candidate].wrapped
        obs = torch.as_tensor(observation, dtype=torch.float32, device=owner.device)[
            None
        ]
        return owner.actor.get_action(obs)[2][0].cpu().numpy()
    return agent.eval(observation, np.asarray(candidate, np.float32))


def modules(agent, method):
    if method == "momba":
        return {"agent": agent.impl}
    result = {}
    owners = (
        [p.wrapped for p in agent.population]
        if hasattr(agent, "population")
        else [agent]
    )
    for i, owner in enumerate(owners):
        for name in ("policy", "actor", "critic", "qf1", "qf2", "duals", "q_nets"):
            value = getattr(owner, name, None)
            if isinstance(value, torch.nn.Module):
                result[f"policy{i}.{name}"] = value
            elif isinstance(value, (list, tuple)):
                for j, network in enumerate(value):
                    result[f"policy{i}.{name}.{j}"] = network
    if method == "morld_mosac":
        for i, candidate in enumerate(agent.archive.individuals):
            result[f"archive{i}.actor"] = candidate.wrapped.actor
    return result
