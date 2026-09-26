# MuJoCo critic architectures

Five multi-objective reinforcement-learning methods with shared, multi-head,
and independent objective critics. One runner supports six two-objective tasks.

## Install

Use Python 3.12. On Ubuntu/Debian, the GPI-LS dependency needs GMP headers
and a compiler: `sudo apt install build-essential python3-dev libgmp-dev`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python prepare.py
```

`prepare.py` downloads six checksum-verified files from the pinned public
Momba repository. Dependencies are fetched during installation; the training
runner uses local logging and disables external experiment tracking.

## Run

```bash
python train.py --env hopper --method momba --architecture split --seed 0
```

Change `--env` to `ant`, `halfcheetah`, `walker2d`, `hopper`, `swimmer`, or
`humanoid`. Change `--method` to `momba`, `capql`, `morld_mosac`, `mo_mpo`,
or `gpi_ls`. Architectures are `shared`, `multihead`, and `split`.

```bash
# Another method and architecture.
python train.py --env ant --method capql --architecture multihead --out results/ant

# Quick CPU execution check; not a performance experiment.
python train.py --env swimmer --method momba --smoke --device cpu --out results/check
```

The default is 200,000 training transitions, with evaluation every 10,000
transitions, five episodes per candidate policy, and 101 utility queries.
The five-policy methods share this transition budget across their policies.
GPI-LS retains native early stopping. `--device auto` uses an available GPU,
otherwise CPU. Use a different `--out` directory for each run; existing
directories are never overwritten.

Only local numeric outputs are saved: `config.json`, `metrics.jsonl`,
`front_<step>.npz`, and `model.pt`. The model file stores inference weights,
not a training-resume snapshot. Front files contain candidate-by-episode
reward vectors. Configuration files contain algorithm options only.

## Architecture details

- **Shared:** one shared hidden path and vector output. For Momba, this option
  is the native **scalarized-value** critic, not a shared vector critic.
- **Multi-head:** shared first hidden block with objective-specific later
  layers. Momba shares its Simba trunk; MO-MPO uses its native multi-head model.
- **Split:** a complete critic path for each objective, with no shared critic
  parameters across objectives. The actor is unchanged.

Momba keeps observation/reward normalization, a distributional critic, and
automatic entropy tuning. `--twin-selection objective` optionally changes
its vector critics from group selection to objective-wise minima.
MORL/D + MOSAC retains its scalarized TD loss; separate branches do not change
that supervision. MO-MPO uses TD(0). GPI-LS is model-free and retains its
native LayerNorm, dropout, prioritized replay, and linear-support mechanism.

Optional `--actor-lr`, `--critic-lr`, and `--critic-width` override the default
settings. Changing critic width leaves the actor fixed. The MLP critic path
uses `(h,h)`; MO-MPO uses `(h,h,round(0.75*h))`; Momba uses Simba hidden size
`h`. Setting equal widths does not match total parameter counts across
architectures.

See `THIRD_PARTY.md` for implementation sources.
