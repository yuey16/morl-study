# Implementation sources

- CAPQL, MORL/D + MOSAC, and GPI-LS use
  [MORL-Baselines](https://github.com/LucasAlegre/morl-baselines), pinned in
  `requirements.txt`. Its upstream MIT license and notices remain with the
  installed dependency.
- Momba uses [the public Momba implementation](https://github.com/adamstafa/momba).
  `prepare.py` downloads the exact upstream files listed in `sources.json`;
  those files are not bundled in this source archive. The local code adds
  objective-specific critic branches and a small PyTorch training loop.
- MO-MPO is a TD(0) implementation of the per-objective E-step and Gaussian
  policy projection described in
  [Abdolmaleki et al., ICML 2020](https://proceedings.mlr.press/v119/abdolmaleki20a.html).
- MuJoCo environment implementations come from
  [MO-Gymnasium](https://github.com/Farama-Foundation/MO-Gymnasium).

- `grid/src/residential_grid_morl` contains the supplied residential-grid
  environment source; the included `v4/ieee33_pv_rich.yaml` configuration selects
  the five-objective battery/renewable case. Its numerical dynamics are retained.
  IEEE33 network data is supplied by pandapower. Processed input provenance is
  documented in `grid/DATA.md`; original dataset terms continue to apply.

Dependency licenses are not replaced by this package.
