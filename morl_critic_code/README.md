# Multi-objective critic architectures

Five multi-objective reinforcement-learning methods with shared, multi-head,
and independent objective critics. One runner supports six two-objective MuJoCo tasks and the five-objective
IEEE33 `pv_rich` distribution-grid task.

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
`humanoid`, or `ieee33_pv_rich`. Change `--method` to `momba`, `capql`, `morld_mosac`, `mo_mpo`,
or `gpi_ls`. Architectures are `shared`, `multihead`, and `split`.

```bash
# Another method and architecture.
python train.py --env ant --method capql --architecture multihead --out results/ant

# Quick CPU execution check; not a performance experiment.
python train.py --env swimmer --method momba --smoke --device cpu --out results/check
```

For MuJoCo, the default is 200,000 training transitions, with evaluation every 10,000
transitions, five episodes per candidate policy, and 101 utility queries.
Population methods share the transition budget across their policies.
GPI-LS retains native early stopping. `--device auto` uses an available GPU,
otherwise CPU. Use a different `--out` directory for each run; existing
directories are never overwritten.

## Architecture details

- **Shared:** 
- **Multi-head:** .
- **Split:** 

Optional `--actor-lr`, `--critic-lr`, and `--critic-width` override the default
settings. Changing critic width leaves the actor fixed. The MLP critic path
uses `(h,h)`; MO-MPO uses `(h,h,round(0.75*h))`; Momba uses Simba hidden size
`h`. Setting equal widths does not match total parameter counts across
architectures.

## IEEE33: five objectives

```bash
python train.py --env ieee33_pv_rich --method momba --architecture split --seed 0 --out results/ieee33

# Short execution check, including evaluation and model saving.
python train.py --env ieee33_pv_rich --method capql --architecture multihead --smoke --device cpu --out results/grid_check
```

The included case is **IEEE33 `pv_rich`, battery/renewable five-objective mode**.
The objectives minimize electricity cost, EV service deficit, voltage penalties,
battery degradation, and renewable non-utilisation. Rewards are their negatives.
Actions control four batteries, 440 EV ports, and 32 PV-curtailment fractions:
476 action components, 2,712 observation components, and 96 fifteen-minute steps
per daily episode. The AC power-flow model uses pandapower's IEEE33 feeder.

The runner maps actions from `[-1,1]` to the environment's physical bounds and
divides each reward component by the fixed scale in `envs.py`. Training episodes
use the training split. The four non-Momba methods use running observation
statistics updated only from training observations; replay stores raw inputs.
Momba retains its own observation and reward normalization. Daily endings are
terminal transitions in this finite-horizon task.

IEEE33 defaults to **500,000 training transitions**. Validation runs every
10,000 transitions with five episodes per candidate. Evaluation uses 35 fixed
preferences and 1,005 fixed utility queries. MORL/D evaluates up to 35 archived
policies; MO-MPO uses seven policies sharing the total transition budget.
The final model is also evaluated on 30 fixed test scenarios, saved separately
as `test_metrics.json` and `test_front.npz`. Learning curves contain validation
scores. `--smoke` uses 100 training transitions, 25-step episodes and one
validation/test episode; its scores are only execution checks. The grid GPI-LS
smoke check performs one linear-support training iteration.

The environment source, configuration and required processed inputs are under
`grid/`. No separate dataset download is needed. Only the IEEE33 case is packaged.
See `grid/DATA.md` for input provenance. The package includes algorithms and
runnable defaults, without experiment histories or selected-result tables.
Full training requires substantial RAM for replay because observations and
actions are large.

See `THIRD_PARTY.md` for implementation sources.
