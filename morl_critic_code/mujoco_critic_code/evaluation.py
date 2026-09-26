"""Local evaluation without changing the training random-number streams."""

from contextlib import contextmanager
from copy import deepcopy
import random

import numpy as np
import torch

from algorithms.baselines import action, candidates
from envs import Environment, preferences


@contextmanager
def isolated_randomness(agent):
    cpu = torch.get_rng_state()
    gpu = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    python, numpy = random.getstate(), np.random.get_state()
    owners = [agent] + [p.wrapped for p in getattr(agent, "population", [])]
    generators = [
        (value, deepcopy(value.bit_generator.state))
        for owner in owners
        for value in vars(owner).values()
        if isinstance(value, np.random.Generator)
    ]
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if gpu is not None:
            torch.cuda.set_rng_state_all(gpu)
        random.setstate(python)
        np.random.set_state(numpy)
        for generator, state in generators:
            generator.bit_generator.state = state


def evaluate(agent, args):
    options = candidates(agent, args.method, 3 if args.smoke else 11)
    if not options:
        return None
    env = Environment(args.env, args.seed + 10000, args.smoke)
    returns = np.zeros((len(options), 1 if args.smoke else args.eval_episodes, 2))
    try:
        with isolated_randomness(agent):
            for i, candidate in enumerate(options):
                for episode in range(returns.shape[1]):
                    obs, _ = env.reset(seed=args.seed + 10000 + episode)
                    done = False
                    while not done:
                        obs, reward, terminal, truncated, _ = env.step(
                            action(agent, args.method, obs, candidate)
                        )
                        returns[i, episode] += reward
                        done = terminal or truncated
    finally:
        env.close()
    means = returns.mean(axis=1)
    if not np.isfinite(returns).all():
        raise FloatingPointError("Non-finite evaluation returns.")
    nondominated = [
        i
        for i, point in enumerate(means)
        if not np.any(np.all(means >= point, axis=1) & np.any(means > point, axis=1))
    ]
    eum = (preferences(101) @ means.T).max(axis=1).mean()
    return dict(
        eum=float(eum),
        returns=means.tolist(),
        front=means[nondominated].tolist(),
        evaluation_steps=env.steps,
    ), returns
