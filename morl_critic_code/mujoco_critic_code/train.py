"""Run one algorithm and critic architecture on one MuJoCo environment."""

import argparse
import json
import os
from pathlib import Path
import random

# The public baseline dependency imports W&B; all learners also use log=False.
os.environ["WANDB_MODE"] = "disabled"

import numpy as np
import torch

from algorithms.baselines import build, modules, train
from envs import ENVIRONMENTS, Environment
from evaluation import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENVIRONMENTS, default="hopper")
    parser.add_argument(
        "--method",
        choices=["momba", "capql", "morld_mosac", "mo_mpo", "gpi_ls"],
        default="momba",
    )
    parser.add_argument(
        "--architecture", choices=["shared", "multihead", "split"], default="split"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out", default="results/run")
    parser.add_argument(
        "--eval-every", type=int, default=10000, help="0 evaluates only at the end"
    )
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument(
        "--critic-width", type=int, help="change only critic width; actor stays fixed"
    )
    parser.add_argument("--actor-lr", type=float)
    parser.add_argument("--critic-lr", type=float)
    parser.add_argument(
        "--twin-selection",
        choices=["group", "objective"],
        default="group",
        help="Momba only",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="100-step execution check with small batches and short episodes",
    )
    args = parser.parse_args()
    if args.smoke:
        args.steps, args.eval_every = 100, 50
    if args.steps < 1 or args.eval_every < 0 or args.eval_episodes < 1:
        parser.error(
            "Steps and episode counts must be positive; eval-every must be nonnegative."
        )
    if args.critic_width is not None and args.critic_width < 2:
        parser.error("Critic width must be at least 2.")
    if any(
        x is not None and (not np.isfinite(x) or x <= 0)
        for x in (args.actor_lr, args.critic_lr)
    ):
        parser.error("Learning rates must be finite and positive.")
    if args.twin_selection == "objective" and (
        args.method != "momba" or args.architecture == "shared"
    ):
        parser.error("Objective twin selection applies only to Momba multihead/split.")
    block = (
        (20 if args.smoke else 10000)
        if args.method in ("morld_mosac", "gpi_ls")
        else 5
        if args.method == "mo_mpo"
        else 1
    )
    if args.steps % block:
        parser.error(f"This method requires steps divisible by {block}.")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA/ROCm is unavailable; use --device cpu.")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = {
        key: value for key, value in vars(args).items() if key not in ("out", "device")
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    env = Environment(args.env, args.seed, args.smoke)
    selection = Environment(args.env, args.seed + 20000, args.smoke)
    agent = build(env, args)
    last_evaluation = -1

    def record(step):
        nonlocal last_evaluation
        result = evaluate(agent, args)
        if result is None:
            return
        metric, returns = result
        metric["step"] = step
        with (out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metric, allow_nan=False) + "\n")
        np.savez_compressed(out / f"front_{step}.npz", returns=returns)
        print(f"step={step} eum={metric['eum']:.4f}", flush=True)
        last_evaluation = step

    def before_step(step):
        if (
            args.eval_every
            and step > 0
            and step % args.eval_every == 0
            and step != last_evaluation
        ):
            record(step)

    env.before_step = before_step
    try:
        train(agent, env, selection, args)
        if last_evaluation != env.steps:
            record(env.steps)
        networks = modules(agent, args.method)
        if not all(
            torch.isfinite(p).all()
            for network in networks.values()
            for p in network.parameters()
        ):
            raise FloatingPointError("Non-finite model parameters.")
        checkpoint = {
            key: {
                name: tensor.detach().cpu()
                for name, tensor in network.state_dict().items()
            }
            for key, network in networks.items()
        }
        if args.method == "gpi_ls":
            checkpoint["weight_support"] = [
                torch.as_tensor(weight).detach().cpu()
                for weight in agent.weight_support
            ]
        if args.method == "morld_mosac":
            checkpoint["archive_weights"] = torch.tensor(
                np.asarray([p.weights for p in agent.archive.individuals])
            )
        torch.save(checkpoint, out / "model.pt")
        print(f"Finished: {env.steps} training transitions.")
    finally:
        env.close()
        selection.close()


if __name__ == "__main__":
    main()
